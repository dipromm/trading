"""
El Juez Central v1 — Ensemble Estático Ponderado (Meta-Modelo).

Recibe los outputs calibrados de los agentes y los combina para emitir
la decisión final de inversión por ticker y fecha.

Evolución del Juez a lo largo del proyecto:
    - Fase 3 (MVP):  pass-through del Matemático (sin entrenamiento)
    - Fase 4:        meta-modelo entrenado con [matemático, analista]
    - Fase 5+:       meta-modelo con todos los agentes
    - Fase 6 (opt):  Juez v2 con RL/PPO (ver judge/judge_v2_rl.py)

IMPORTANTE — El Conspiranoico tiene poder de VETO:
    Si veto_signal[fecha] == 1, el Juez retorna señal 0 para todos los tickers
    ese día, independientemente de lo que digan los demás agentes.

IMPORTANTE — Entrenamiento del Juez v1:
    El Juez no puede entrenarse con las predicciones de los agentes sobre los
    mismos datos que usó para entrenarse (data leakage). Se entrena con las
    predicciones de los agentes de la iteración anterior del walk-forward.
    En la Iteración 1, el Juez es simple pass-through.

Dos formatos de entrada soportados (ver ``predict``):

    Formato "multi-ticker" (usado por el BacktestEngine):
        agent_predictions es un DataFrame donde columnas = tickers y
        valores = señal de un único agente (el Matemático en Fase 3).
        Retorna un DataFrame del mismo shape con la señal final.

    Formato "multi-agente" (Fase 4+, meta-modelo):
        agent_predictions es un DataFrame donde columnas = agentes
        (e.g. "matematico", "analista") y filas = observaciones.
        El Juez aplica el meta-modelo entrenado y retorna una Series.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from utils.config_loader import load_config

logger = logging.getLogger(__name__)

_MIN_FIT_ROWS = 60
_PASS_THROUGH_COLUMN = "matematico"


class JuezV1:
    """
    Meta-modelo que combina las señales de los agentes.

    Modos de operación:
        * **Pass-through** (``is_pass_through=True``, por defecto):
          no requiere ``fit()``. ``predict()`` simplemente retorna las
          señales del Matemático sin modificarlas. Válido para Fase 3.

        * **Meta-modelo** (``is_pass_through=False``, tras ``fit()``):
          un clasificador calibrado (Logistic o XGBoost) entrenado sobre
          las señales combinadas de múltiples agentes. Válido desde Fase 4.

    En ambos modos, ``predict()`` aplica el veto del Conspiranoico si se
    proporciona.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        cfg = config["judge_v1"]

        self._meta_model_type: str = cfg["meta_model"]
        self._calibration_method: str = cfg["calibration_method"]
        self._random_state: int = cfg["random_state"]
        self._cv_folds: int = cfg.get("calibration_cv_folds", 5)

        self._build_model()
        self._is_pass_through: bool = True
        self._is_fitted: bool = False
        self._agent_columns: list[str] = []
        self._n_fit_rows: int = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_model(self) -> None:
        """(Re)construye el meta-modelo calibrado."""
        if self._meta_model_type == "logistic":
            base = LogisticRegression(
                random_state=self._random_state,
                max_iter=1000,
            )
        else:
            base = XGBClassifier(
                random_state=self._random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            )

        self.model = CalibratedClassifierCV(
            estimator=base,
            method=self._calibration_method,
            cv=self._cv_folds,
        )

    @staticmethod
    def _apply_veto(
        result: pd.DataFrame | pd.Series,
        veto_signal: pd.Series | None,
    ) -> pd.DataFrame | pd.Series:
        """Pone a 0 todas las señales en fechas con veto activo."""
        if veto_signal is None:
            return result

        veto_dates = veto_signal.index[veto_signal == 1]
        common = result.index.intersection(veto_dates)

        if common.empty:
            return result

        if isinstance(result, pd.DataFrame):
            result.loc[common, :] = 0.0
        else:
            result.loc[common] = 0.0

        if len(common) > 0:
            logger.info(
                "Veto del Conspiranoico aplicado en %d fechas", len(common),
            )
        return result

    # ── Interfaz pública ──────────────────────────────────────────────────────

    @property
    def is_pass_through(self) -> bool:
        return self._is_pass_through

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        agent_predictions: pd.DataFrame,
        targets: pd.Series,
    ) -> None:
        """
        Entrena el meta-modelo con las predicciones de los agentes.

        Tras ``fit()``, el Juez pasa del modo pass-through al modo
        meta-modelo. ``predict()`` usará el modelo entrenado en lugar
        de reenviar directamente la señal del Matemático.

        Anti-leakage: ``agent_predictions`` y ``targets`` deben provenir
        de un período de walk-forward **anterior** al período de
        validación actual.

        Args:
            agent_predictions: DataFrame donde cada columna es la señal
                de un agente (e.g. "matematico", "analista", "cazador").
                Filas = observaciones (fecha × ticker del período anterior).
            targets: Serie binaria alineada con ``agent_predictions``
                (1 = el precio subió, 0 = bajó).

        Raises:
            ValueError: Si hay menos de ``_MIN_FIT_ROWS`` filas limpias
                o si ``agent_predictions`` tiene menos de 2 columnas
                (con un solo agente, pass-through es más apropiado).
        """
        if agent_predictions.shape[1] < 2:
            raise ValueError(
                f"El meta-modelo necesita señales de al menos 2 agentes, "
                f"pero se recibieron {agent_predictions.shape[1]} columna(s). "
                f"Con un solo agente, usa el modo pass-through (no llames a fit)."
            )

        combined = agent_predictions.copy()
        combined["_target"] = targets
        clean = combined.dropna()

        if len(clean) < _MIN_FIT_ROWS:
            raise ValueError(
                f"Solo {len(clean)} filas limpias tras eliminar NaN "
                f"(mínimo requerido: {_MIN_FIT_ROWS}). "
                f"Necesitas más datos del período anterior de walk-forward."
            )

        X = clean.drop(columns=["_target"])
        y = clean["_target"]

        self._agent_columns = list(X.columns)

        self._build_model()
        self.model.fit(X, y)
        self._is_pass_through = False
        self._is_fitted = True
        self._n_fit_rows = len(clean)

        logger.info(
            "Juez v1 entrenado — tipo: %s, agentes: %s, filas: %d, "
            "calibración: %s (cv=%d)",
            self._meta_model_type,
            self._agent_columns,
            self._n_fit_rows,
            self._calibration_method,
            self._cv_folds,
        )

    def predict(
        self,
        agent_predictions: pd.DataFrame,
        veto_signal: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Emite la señal final combinando las predicciones de los agentes.

        Soporta dos formatos de entrada según la fase del proyecto:

        **Formato multi-ticker (Fase 3 pass-through):**
            ``agent_predictions`` tiene columnas = tickers, index = fechas,
            valores = señal del Matemático. El Juez retorna el mismo
            DataFrame (aplicando veto si corresponde).

        **Formato multi-agente (Fase 4+ meta-modelo):**
            ``agent_predictions`` tiene columnas = nombres de agentes
            (deben coincidir con las columnas usadas en ``fit()``),
            index = observaciones. El Juez aplica el meta-modelo y
            retorna un DataFrame con una columna ``"prob_up"`` con la
            probabilidad calibrada combinada.

        Args:
            agent_predictions: DataFrame con señales de agentes.
            veto_signal: Serie del Conspiranoico indexada por fecha.
                1 = veto activo (forzar señal 0), 0 = mercado normal.
                Si ``None``, no se aplica veto.

        Returns:
            DataFrame con la señal final ∈ [0, 1].
            Fechas con veto activo tendrán valor 0.0.
        """
        if self._is_pass_through:
            result = agent_predictions.copy().astype(float)
            return self._apply_veto(result, veto_signal)

        if not self._is_fitted:
            raise RuntimeError(
                "El Juez no está en modo pass-through ni ha sido entrenado. "
                "Llama a fit() o usa el modo pass-through (Fase 3)."
            )

        missing = [c for c in self._agent_columns if c not in agent_predictions.columns]
        if missing:
            raise ValueError(
                f"Faltan columnas de agentes que se usaron en fit(): {missing}. "
                f"Columnas disponibles: {list(agent_predictions.columns)}"
            )

        X = agent_predictions[self._agent_columns].copy()
        clean_mask = X.notna().all(axis=1)
        X_clean = X.loc[clean_mask]

        result = pd.DataFrame(
            np.nan,
            index=agent_predictions.index,
            columns=["prob_up"],
        )

        if not X_clean.empty:
            proba = self.model.predict_proba(X_clean)[:, 1]
            result.loc[X_clean.index, "prob_up"] = proba

        return self._apply_veto(result, veto_signal)

    def predict_single_agent(
        self,
        signal: pd.Series,
        veto_signal: pd.Series | None = None,
    ) -> pd.Series:
        """
        Conveniencia para Fase 3: pasa la señal de un agente directamente.

        Equivalente a ``predict(signal.to_frame(), veto_signal)`` pero
        retorna una Series en lugar de un DataFrame.

        Args:
            signal: Serie de probabilidades del Matemático indexada por fecha.
            veto_signal: Serie del Conspiranoico (opcional).

        Returns:
            Serie de probabilidades finales ∈ [0, 1].
        """
        result = signal.copy().astype(float)
        return self._apply_veto(result, veto_signal)

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    def evaluation_report(
        self,
        agent_predictions: pd.DataFrame,
        targets: pd.Series,
    ) -> dict[str, Any]:
        """
        Evalúa el meta-modelo sobre un período de validación.

        Solo funciona en modo meta-modelo (tras ``fit()``). Para Fase 3
        (pass-through), evaluar directamente las métricas del Matemático.

        Args:
            agent_predictions: Señales de agentes del período de validación.
            targets: Resultados reales binarios alineados.

        Returns:
            dict con Brier Score, Log Loss y número de muestras evaluadas.
        """
        if self._is_pass_through:
            raise RuntimeError(
                "evaluation_report() no aplica en modo pass-through. "
                "Evalúa directamente el agente con calibration_report()."
            )
        if not self._is_fitted:
            raise RuntimeError("Llama a fit() antes de evaluation_report().")

        X = agent_predictions[self._agent_columns].copy()
        combined = pd.DataFrame({"_target": targets}, index=X.index)
        for col in X.columns:
            combined[col] = X[col]
        clean = combined.dropna()

        if clean.empty:
            raise ValueError("No hay filas limpias para evaluar.")

        X_clean = clean[self._agent_columns]
        y_true = clean["_target"].values
        y_prob = self.model.predict_proba(X_clean)[:, 1]

        report = {
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 5),
            "log_loss": round(float(log_loss(y_true, y_prob)), 5),
            "n_samples": len(clean),
            "agent_columns": self._agent_columns,
            "meta_model_type": self._meta_model_type,
        }

        logger.info(
            "Juez v1 evaluación — Brier: %.4f, LogLoss: %.4f, muestras: %d",
            report["brier_score"], report["log_loss"], report["n_samples"],
        )
        return report

    def agent_weights(self) -> pd.Series | None:
        """
        Retorna los pesos/coeficientes del meta-modelo para cada agente.

        Solo disponible para ``meta_model='logistic'``. Para XGBoost,
        retorna ``feature_importances_`` promediadas de los calibradores.

        Returns:
            ``pd.Series`` indexada por nombre de agente, o ``None`` en
            modo pass-through.
        """
        if self._is_pass_through or not self._is_fitted:
            return None

        if self._meta_model_type == "logistic":
            coefs_list: list[np.ndarray] = []
            for calibrated_clf in self.model.calibrated_classifiers_:
                coefs_list.append(calibrated_clf.estimator.coef_[0])
            mean_coefs = np.mean(coefs_list, axis=0)
            return pd.Series(
                mean_coefs, index=self._agent_columns, name="weight",
            ).sort_values(ascending=False)

        importances_list: list[np.ndarray] = []
        for calibrated_clf in self.model.calibrated_classifiers_:
            importances_list.append(
                calibrated_clf.estimator.feature_importances_,
            )
        mean_imp = np.mean(importances_list, axis=0)
        return pd.Series(
            mean_imp, index=self._agent_columns, name="importance",
        ).sort_values(ascending=False)
