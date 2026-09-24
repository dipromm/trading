"""
El Conspiranoico — Detección de régimen de mercado anómalo.

Vota VETO (1) cuando detecta un entorno de alta incertidumbre en el que
el resto de agentes no son fiables. Su voto es binario: operar / no operar.

Cuando el veto está activo, el Juez no emite órdenes de compra
independientemente de lo que digan los demás agentes.

Features usadas (todas disponibles con datos diarios de yfinance):
    - Volatilidad realizada rolling (std de retornos a 5, 20, 60 días)
    - VIX (^VIX): índice de miedo del mercado
    - Correlación rolling entre acciones del universo (alta correlación = pánico)
    - Volumen relativo vs media de 20 días
    - Amplitud de mercado: % de acciones del universo que suben ese día

Tecnología: Isolation Forest y/o Gaussian HMM (hmmlearn) + StandardScaler (solo train)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from mas.agents.base_agent import AgentBase
from mas.utils.config_loader import load_config

logger = logging.getLogger(__name__)

DETECTOR_ISOLATION_FOREST = "isolation_forest"
DETECTOR_HMM = "hmm"
DETECTOR_HYBRID = "hybrid"
VALID_DETECTORS = {DETECTOR_ISOLATION_FOREST, DETECTOR_HMM, DETECTOR_HYBRID}


class Conspiranoico(AgentBase):
    """
    Agente de detección de régimen de mercado anómalo.
    Emite veto cuando detecta condiciones de alta incertidumbre.

    A diferencia de los agentes predictivos (Matemático, Analista, Cazador),
    el Conspiranoico opera a nivel de mercado, no por ticker. Su output es
    una Serie binaria por fecha (no una probabilidad por ticker).

    El detector se elige con ``conspiranoico.detector``:
        - isolation_forest: anomaly score + percentiles (default)
        - hmm: Gaussian HMM con estados ordenados por VIX+vol
        - hybrid: min(escala_IF, escala_HMM) — más conservador

    El Isolation Forest es un modelo no supervisado: no necesita target.
    Se entrena sobre la distribución de los regímenes normales; los días
    que se desvían mucho de esa distribución reciben un anomaly score bajo
    (más negativo). El umbral de veto se fija en el percentil indicado por
    veto_threshold_percentile sobre los scores del período de entrenamiento.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        cfg = config["conspiranoico"]

        self.model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            random_state=cfg["random_state"],
        )
        self._scaler = StandardScaler()
        self._veto_threshold_percentile: int = cfg.get("veto_threshold_percentile", 5)
        self._veto_mode: str = cfg.get("veto_mode", "binary")
        self._soft_scale_severe: float = float(cfg.get("soft_veto_scale_severe", 0.25))
        self._soft_scale_moderate: float = float(cfg.get("soft_veto_scale_moderate", 0.6))
        self._soft_moderate_percentile: int = int(cfg.get("soft_veto_moderate_percentile", 15))
        self._detector: str = cfg.get("detector", DETECTOR_ISOLATION_FOREST)
        if self._detector not in VALID_DETECTORS:
            raise ValueError(
                f"conspiranoico.detector debe ser uno de {sorted(VALID_DETECTORS)}, "
                f"recibido: {self._detector!r}"
            )

        hmm_cfg = cfg.get("hmm", {})
        self._hmm_n_states: int = int(hmm_cfg.get("n_states", 3))
        self._hmm_n_iter: int = int(hmm_cfg.get("n_iter", 100))
        self._hmm_covariance_type: str = hmm_cfg.get("covariance_type", "diag")
        self._hmm_veto_min_risk_rank: Optional[int] = hmm_cfg.get("veto_min_risk_rank")
        self._hmm: Optional[GaussianHMM] = None
        self._state_risk_rank: dict[int, int] = {}
        self._crisis_state: Optional[int] = None

        self._veto_threshold: Optional[float] = None
        self._moderate_threshold: Optional[float] = None
        self._is_fitted = False

    # ── Interfaz AgentBase ────────────────────────────────────────────────────

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entrena el Isolation Forest sobre las features de régimen del período
        de entrenamiento.

        Args:
            train_data: DataFrame con las features devueltas por
                data.regime.build_regime_features(), sliced al período train.
                Las columnas esperadas son: vol_5d, vol_20d, vol_60d, vix,
                avg_correlation, avg_volume_ratio, market_breadth.

        Raises:
            ValueError: Si quedan menos de 20 filas limpias tras eliminar NaN.
        """
        clean = train_data.dropna()
        if len(clean) < 20:
            raise ValueError(
                f"Solo {len(clean)} filas limpias en train_data "
                f"(mínimo 20). Verifica que build_regime_features() "
                f"tiene datos suficientes para la ventana de entrenamiento."
            )

        X = clean.values

        # Scaler ajustado únicamente en train (anti-leakage)
        X_scaled = self._scaler.fit_transform(X)

        use_if = self._detector in (DETECTOR_ISOLATION_FOREST, DETECTOR_HYBRID)
        use_hmm = self._detector in (DETECTOR_HMM, DETECTOR_HYBRID)

        train_scores: Optional[np.ndarray] = None

        if use_if:
            self.model.fit(X_scaled)
            train_scores = self.model.score_samples(X_scaled)
            self._veto_threshold = float(
                np.percentile(train_scores, self._veto_threshold_percentile)
            )
            if self._veto_mode == "soft":
                if self._soft_moderate_percentile <= self._veto_threshold_percentile:
                    raise ValueError(
                        f"soft_veto_moderate_percentile ({self._soft_moderate_percentile}) "
                        f"debe ser mayor que veto_threshold_percentile "
                        f"({self._veto_threshold_percentile})"
                    )
                self._moderate_threshold = float(
                    np.percentile(train_scores, self._soft_moderate_percentile)
                )
            else:
                self._moderate_threshold = None

        if use_hmm:
            self._fit_hmm(X_scaled, clean)

        if use_if and train_scores is not None:
            n_anomalous_train = int((train_scores < self._veto_threshold).sum())
            pct_anomalous = n_anomalous_train / len(train_scores) * 100
            logger.info(
                "Conspiranoico IF — %d filas | umbral=%.4f (p%d) | "
                "anomalías train: %d (%.1f%%)%s",
                len(clean),
                self._veto_threshold,
                self._veto_threshold_percentile,
                n_anomalous_train,
                pct_anomalous,
                (
                    f" | umbral_moderado={self._moderate_threshold:.4f} (p{self._soft_moderate_percentile})"
                    if self._veto_mode == "soft"
                    else ""
                ),
            )

        if use_hmm:
            logger.info(
                "Conspiranoico HMM — %d estados | crisis_state=%d | "
                "ranking riesgo=%s | veto_min_rank=%d",
                self._hmm_n_states,
                self._crisis_state,
                self._state_risk_rank,
                self._effective_veto_min_risk_rank(),
            )

        logger.info(
            "Conspiranoico entrenado — detector=%s | modo=%s | %d filas limpias",
            self._detector,
            self._veto_mode,
            len(clean),
        )

        self._is_fitted = True

    def _fit_hmm(self, X_scaled: np.ndarray, clean: pd.DataFrame) -> None:
        """Entrena Gaussian HMM y ordena estados por riesgo (VIX + vol_20d)."""
        if self._hmm_n_states < 2:
            raise ValueError(f"hmm.n_states debe ser >= 2, recibido {self._hmm_n_states}")

        self._hmm = GaussianHMM(
            n_components=self._hmm_n_states,
            covariance_type=self._hmm_covariance_type,
            n_iter=self._hmm_n_iter,
            random_state=self.model.random_state,
        )
        self._hmm.fit(X_scaled)

        train_states = self._hmm.predict(X_scaled)
        state_risk: dict[int, float] = {}
        vix = clean["vix"] if "vix" in clean.columns else pd.Series(0.0, index=clean.index)
        vol20 = (
            clean["vol_20d"] if "vol_20d" in clean.columns else pd.Series(0.0, index=clean.index)
        )

        for state in range(self._hmm_n_states):
            mask = train_states == state
            if mask.sum() == 0:
                state_risk[state] = float(state)
            else:
                state_risk[state] = float(vix.iloc[mask].mean() + vol20.iloc[mask].mean() * 100.0)

        sorted_states = sorted(state_risk.keys(), key=lambda s: state_risk[s])
        self._state_risk_rank = {state: rank for rank, state in enumerate(sorted_states)}
        self._crisis_state = sorted_states[-1]

    def _effective_veto_min_risk_rank(self) -> int:
        """Rango mínimo de riesgo que activa veto binario en modo HMM."""
        if self._hmm_veto_min_risk_rank is not None:
            return int(self._hmm_veto_min_risk_rank)
        return self._hmm_n_states - 1

    def _hmm_states(self, X_scaled: np.ndarray) -> np.ndarray:
        if self._hmm is None:
            raise RuntimeError("HMM no entrenado")
        return self._hmm.predict(X_scaled)

    def _hmm_risk_ranks(self, X_scaled: np.ndarray) -> np.ndarray:
        states = self._hmm_states(X_scaled)
        return np.array([self._state_risk_rank[int(s)] for s in states])

    def _scale_for_risk_rank(self, rank: int) -> float:
        max_rank = self._hmm_n_states - 1
        if rank <= 0:
            return 1.0
        if rank >= max_rank:
            return self._soft_scale_severe
        if max_rank == 1:
            return self._soft_scale_moderate
        return self._soft_scale_moderate

    def _hmm_risk_scale(self, X_scaled: np.ndarray) -> np.ndarray:
        ranks = self._hmm_risk_ranks(X_scaled)
        return np.array([self._scale_for_risk_rank(int(r)) for r in ranks])

    def _hmm_veto(self, X_scaled: np.ndarray) -> np.ndarray:
        ranks = self._hmm_risk_ranks(X_scaled)
        return (ranks >= self._effective_veto_min_risk_rank()).astype(int)

    def _if_risk_scale(self, scores: np.ndarray) -> np.ndarray:
        scale = np.ones(len(scores), dtype=float)
        severe = scores < self._veto_threshold
        moderate = (~severe) & (scores < self._moderate_threshold)
        scale[severe] = self._soft_scale_severe
        scale[moderate] = self._soft_scale_moderate
        return scale

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna señal de veto por fecha.

        Args:
            data: DataFrame con las mismas columnas que las usadas en fit(),
                  sliced al período de validación.

        Returns:
            Serie binaria: 1 = veto activo (mercado anómalo), 0 = mercado normal.
            Fechas con NaN en features reciben veto=0 (conservador: no vetar
            cuando no hay información).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Conspiranoico no está entrenado. Llama a fit() antes de predict()."
            )

        # Índice completo del período de validación
        result = pd.Series(0, index=data.index, dtype=int, name="conspiranoico_veto")

        clean_mask = data.notna().all(axis=1)
        data_clean = data.loc[clean_mask]

        if data_clean.empty:
            logger.warning(
                "Conspiranoico: no hay filas sin NaN en el período de predicción. "
                "Veto = 0 para todos los días."
            )
            return result

        X_scaled = self._scaler.transform(data_clean.values)

        if self._detector == DETECTOR_ISOLATION_FOREST:
            scores = self.model.score_samples(X_scaled)
            veto = (scores < self._veto_threshold).astype(int)
        elif self._detector == DETECTOR_HMM:
            veto = self._hmm_veto(X_scaled)
        else:
            scores = self.model.score_samples(X_scaled)
            if_veto = (scores < self._veto_threshold).astype(int)
            hmm_veto = self._hmm_veto(X_scaled)
            veto = np.maximum(if_veto, hmm_veto)

        result.loc[data_clean.index] = veto

        n_veto = int(veto.sum())
        pct_veto = n_veto / len(veto) * 100
        logger.info(
            "Conspiranoico predict — %d días evaluados | veto activo: %d (%.1f%%)",
            len(veto), n_veto, pct_veto,
        )

        return result

    def predict_risk_scale(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna multiplicador de Kelly por fecha según anomaly score.

        Solo aplica tiers en ``veto_mode="soft"``. En modo binario retorna 1.0
        siempre (el veto se aplica por separado vía ``predict()``).

        Tiers (scores más negativos = más anómalos):
            - score < umbral p5  → soft_veto_scale_severe (default 0.25)
            - p5 ≤ score < p15   → soft_veto_scale_moderate (default 0.6)
            - score ≥ p15        → 1.0

        Fechas con NaN reciben escala 1.0 (no penalizar sin datos).
        """
        result = pd.Series(1.0, index=data.index, dtype=float, name="conspiranoico_risk_scale")

        if self._veto_mode != "soft":
            if self._detector == DETECTOR_HMM:
                return self._predict_binary_as_scale(data)
            return result

        if not self._is_fitted:
            raise RuntimeError(
                "Conspiranoico no está entrenado. Llama a fit() antes de predict_risk_scale()."
            )

        clean_mask = data.notna().all(axis=1)
        data_clean = data.loc[clean_mask]

        if data_clean.empty:
            logger.warning(
                "Conspiranoico predict_risk_scale: sin filas limpias. Escala = 1.0."
            )
            return result

        X_scaled = self._scaler.transform(data_clean.values)

        if self._detector == DETECTOR_ISOLATION_FOREST:
            scores = self.model.score_samples(X_scaled)
            scale = self._if_risk_scale(scores)
        elif self._detector == DETECTOR_HMM:
            scale = self._hmm_risk_scale(X_scaled)
        else:
            scores = self.model.score_samples(X_scaled)
            if_scale = self._if_risk_scale(scores)
            hmm_scale = self._hmm_risk_scale(X_scaled)
            scale = np.minimum(if_scale, hmm_scale)

        result.loc[data_clean.index] = scale

        severe_mask = scale <= self._soft_scale_severe + 1e-9
        moderate_mask = (~severe_mask) & (scale < 1.0 - 1e-9)
        n_severe = int(severe_mask.sum())
        n_moderate = int(moderate_mask.sum())
        n_normal = len(scale) - n_severe - n_moderate
        logger.info(
            "Conspiranoico risk_scale [%s] — severo=%d (×%.2f) | moderado=%d (×%.2f) | "
            "normal=%d (×1.0)",
            self._detector,
            n_severe, self._soft_scale_severe,
            n_moderate, self._soft_scale_moderate,
            n_normal,
        )

        return result

    def _predict_binary_as_scale(self, data: pd.DataFrame) -> pd.Series:
        """En modo binario + HMM puro, mapea veto a escala 0/1 para stats."""
        veto = self.predict(data)
        return veto.replace({1: self._soft_scale_severe, 0: 1.0}).astype(float)

    @staticmethod
    def summarize_risk_scale(
        risk_scale: pd.Series,
        severe_scale: float = 0.25,
        moderate_scale: float = 0.6,
    ) -> dict[str, int | float]:
        """
        Estadísticas de tiers soft veto para una ventana de validación (Exp8).

        Returns:
            Dict con conteos y % de días severo / moderado / normal / reducido.
        """
        s = risk_scale.dropna()
        n = len(s)
        if n == 0:
            return {
                "n_days": 0,
                "n_severe": 0,
                "n_moderate": 0,
                "n_normal": 0,
                "pct_severe": 0.0,
                "pct_moderate": 0.0,
                "pct_normal": 0.0,
                "pct_reduced": 0.0,
            }

        severe_mask = s <= severe_scale + 1e-9
        moderate_mask = (~severe_mask) & (s < 1.0 - 1e-9)
        n_severe = int(severe_mask.sum())
        n_moderate = int(moderate_mask.sum())
        n_normal = n - n_severe - n_moderate

        return {
            "n_days": n,
            "n_severe": n_severe,
            "n_moderate": n_moderate,
            "n_normal": n_normal,
            "pct_severe": round(n_severe / n * 100, 1),
            "pct_moderate": round(n_moderate / n * 100, 1),
            "pct_normal": round(n_normal / n * 100, 1),
            "pct_reduced": round((n_severe + n_moderate) / n * 100, 1),
        }

    @staticmethod
    def summarize_veto(veto: pd.Series) -> dict[str, int | float]:
        """Estadísticas de veto binario para una ventana (Exp8)."""
        n = len(veto)
        n_veto = int(veto.sum()) if n else 0
        return {
            "n_days": n,
            "n_veto": n_veto,
            "pct_veto": round(n_veto / n * 100, 1) if n else 0.0,
        }

    # ── Método auxiliar para walk-forward ────────────────────────────────────

    def fit_predict_window(
        self,
        regime_features: pd.DataFrame,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
    ) -> pd.Series:
        """
        Encapsula el slice train/val para uso en el walk-forward.

        Args:
            regime_features: DataFrame completo de features de régimen.
            train_start, train_end: Fechas de inicio y fin del entrenamiento.
            val_start, val_end: Fechas de inicio y fin de la validación.

        Returns:
            Serie de veto para el período de validación.
        """
        train_data = regime_features.loc[train_start:train_end]
        val_data = regime_features.loc[val_start:val_end]
        self.fit(train_data)
        return self.predict(val_data)

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    def anomaly_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna los anomaly scores raw (no binarizados) para diagnóstico.

        Scores más negativos = más anómalos. El umbral de veto está en
        self._veto_threshold.

        Args:
            data: DataFrame con las mismas columnas que las usadas en fit().

        Returns:
            Serie de scores por fecha. NaN donde no hay datos limpios.
        """
        if not self._is_fitted:
            raise RuntimeError("Llama a fit() antes de anomaly_scores().")

        result = pd.Series(np.nan, index=data.index, name="anomaly_score")
        clean_mask = data.notna().all(axis=1)
        data_clean = data.loc[clean_mask]

        if data_clean.empty:
            return result

        X_scaled = self._scaler.transform(data_clean.values)

        if self._detector == DETECTOR_ISOLATION_FOREST:
            scores = self.model.score_samples(X_scaled)
        elif self._detector == DETECTOR_HMM:
            proba = self._hmm.predict_proba(X_scaled)
            crisis_idx = self._crisis_state if self._crisis_state is not None else 0
            scores = -proba[:, crisis_idx]
        else:
            if_scores = self.model.score_samples(X_scaled)
            proba = self._hmm.predict_proba(X_scaled)
            crisis_idx = self._crisis_state if self._crisis_state is not None else 0
            hmm_scores = -proba[:, crisis_idx]
            scores = np.minimum(if_scores, hmm_scores)

        result.loc[data_clean.index] = scores
        return result

    @property
    def veto_threshold(self) -> Optional[float]:
        """Umbral de anomaly score por debajo del cual se activa el veto."""
        return self._veto_threshold

    @property
    def veto_mode(self) -> str:
        """Modo de veto: 'binary' o 'soft'."""
        return self._veto_mode

    @property
    def moderate_threshold(self) -> Optional[float]:
        """Umbral p15 para tier moderado (solo modo soft + IF)."""
        return self._moderate_threshold

    @property
    def detector(self) -> str:
        """Backend de detección: isolation_forest, hmm o hybrid."""
        return self._detector

    @property
    def crisis_state(self) -> Optional[int]:
        """Estado HMM con mayor riesgo (solo tras fit con HMM)."""
        return self._crisis_state

    @property
    def state_risk_rank(self) -> dict[int, int]:
        """Mapa estado HMM → rango de riesgo (0=normal, n-1=crisis)."""
        return dict(self._state_risk_rank)
