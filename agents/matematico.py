"""
El Matemático — Análisis técnico con XGBoost + calibración de probabilidades.

Pipeline:
    1. Recibe features técnicas (RSI, MACD, Bollinger, ATR, volumen...)
    2. Entrena un clasificador binario XGBoost: sube (1) / baja (0) mañana
    3. Calibra las probabilidades con Platt Scaling (o Isotonic Regression)
    4. Retorna probabilidad calibrada p ∈ [0, 1] al Juez

La calibración es OBLIGATORIA antes de que el Gestor de Riesgos use Kelly.
Verificar con un reliability diagram después de cada entrenamiento.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from agents.base_agent import AgentBase
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

TARGET_COL = "target_binary"

FEATURE_COLUMNS: list[str] = [
    "close_norm", "return_1d", "return_5d", "return_20d",
    "rsi_14", "rsi_28",
    "macd", "macd_signal", "macd_hist",
    "bb_bandwidth", "bb_pct_b",
    "atr_pct",
    "volume_ratio",
    "sma_20_50_ratio", "sma_50_200_ratio",
]

_MIN_TRAIN_ROWS = 60


class Matematico(AgentBase):
    """
    Agente de análisis técnico basado en XGBoost con probabilidades calibradas.

    Flujo walk-forward por ventana:
        1. ``fit(train_data)``  — entrena XGBoost + calibra probabilidades
        2. ``predict(val_data)`` — retorna p ∈ [0,1] de subida por fila
        3. ``calibration_report(val_data)`` (opcional) — reliability diagram
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        cfg = config["matematico"]

        self._calibration_method: str = cfg["calibration_method"]
        self._cv_folds: int = cfg.get("calibration_cv_folds", 5)

        self._base_params: dict[str, Any] = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "random_state": cfg["random_state"],
            "eval_metric": "logloss",
        }

        self._build_model()
        self._is_fitted = False
        self._train_class_distribution: dict[int, int] = {}
        self._n_train_rows: int = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_model(self) -> None:
        """(Re)construye el pipeline XGBoost + CalibratedClassifierCV."""
        base_model = XGBClassifier(**self._base_params)
        self.model = CalibratedClassifierCV(
            estimator=base_model,
            method=self._calibration_method,
            cv=self._cv_folds,
        )

    @staticmethod
    def _validate_features(df: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Faltan features en el DataFrame: {missing}. "
                f"Asegúrate de pasar el resultado de compute_all_features()."
            )

    @staticmethod
    def _clean(df: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        cols = list(FEATURE_COLUMNS)
        if require_target:
            cols.append(TARGET_COL)
        return df.dropna(subset=cols)

    # ── Interfaz pública (AgentBase) ──────────────────────────────────────────

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entrena el clasificador binario y calibra las probabilidades.

        CalibratedClassifierCV hace cross-validation interna: en cada fold,
        entrena XGBoost en una parte y ajusta la curva de calibración
        (Platt / Isotonic) en la otra. El resultado es un ensemble de
        calibradores cuyo ``predict_proba`` ya devuelve probabilidades
        calibradas.

        Args:
            train_data: DataFrame con ``FEATURE_COLUMNS`` + ``target_binary``.
                        Solo datos del período de entrenamiento walk-forward.

        Raises:
            ValueError: Si faltan features o hay menos de ``_MIN_TRAIN_ROWS``
                        filas limpias tras eliminar NaN.
        """
        self._validate_features(train_data)

        if TARGET_COL not in train_data.columns:
            raise ValueError(
                f"Columna '{TARGET_COL}' no encontrada en train_data. "
                "Genera el target con compute_all_features()."
            )

        clean = self._clean(train_data, require_target=True)

        if len(clean) < _MIN_TRAIN_ROWS:
            raise ValueError(
                f"Solo {len(clean)} filas limpias tras eliminar NaN "
                f"(mínimo requerido: {_MIN_TRAIN_ROWS}). "
                "Verifica que la ventana de entrenamiento es suficientemente larga."
            )

        X = clean[FEATURE_COLUMNS]
        y = clean[TARGET_COL]

        self._build_model()
        self.model.fit(X, y)
        self._is_fitted = True

        self._n_train_rows = len(clean)
        self._train_class_distribution = y.value_counts().to_dict()

        logger.info(
            "Matemático entrenado — filas: %d, clases: %s, "
            "calibración: %s (cv=%d)",
            self._n_train_rows,
            self._train_class_distribution,
            self._calibration_method,
            self._cv_folds,
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna la probabilidad calibrada de subida para cada fila.

        Args:
            data: DataFrame con ``FEATURE_COLUMNS``. Puede contener NaN en
                  algunas filas; esas filas recibirán ``NaN`` como predicción.

        Returns:
            ``pd.Series`` de probabilidades p ∈ [0, 1] indexada como *data*.
            p > 0.5 → señal alcista; p < 0.5 → señal bajista.

        Raises:
            RuntimeError: Si el modelo no ha sido entrenado (``fit`` no llamado).
        """
        if not self.is_fitted():
            raise RuntimeError(
                "Matematico.predict() llamado sin entrenar primero. Llama a fit()."
            )
        self._validate_features(data)

        result = pd.Series(np.nan, index=data.index, name="prob_up")

        clean = self._clean(data, require_target=False)
        if clean.empty:
            logger.warning("predict() recibió datos donde todas las filas tienen NaN en features.")
            return result

        X = clean[FEATURE_COLUMNS]
        proba = self.model.predict_proba(X)[:, 1]
        result.loc[clean.index] = proba

        return result

    # ── Diagnóstico de calibración ────────────────────────────────────────────

    def calibration_report(
        self,
        data: pd.DataFrame,
        n_bins: int = 10,
    ) -> dict[str, Any]:
        """
        Genera métricas de calibración y los datos del reliability diagram.

        Usar después de ``predict()`` sobre el período de *validación*
        para verificar que las probabilidades son fiables antes de pasarlas
        al Gestor de Riesgos / Kelly.

        Args:
            data: DataFrame con ``FEATURE_COLUMNS`` + ``target_binary``
                  (datos de validación walk-forward).
            n_bins: Número de bins para el reliability diagram.

        Returns:
            dict con:
              - ``brier_score``: Brier Score Loss (menor = mejor, 0 = perfecto)
              - ``log_loss``: Log loss (menor = mejor)
              - ``fraction_of_positives``: array — eje Y del reliability diagram
              - ``mean_predicted_value``: array — eje X del reliability diagram
              - ``n_samples``: filas evaluadas
        """
        if not self.is_fitted():
            raise RuntimeError("Llama a fit() antes de calibration_report().")

        self._validate_features(data)
        if TARGET_COL not in data.columns:
            raise ValueError(f"'{TARGET_COL}' requerido para calibration_report().")

        clean = self._clean(data, require_target=True)
        if clean.empty:
            raise ValueError("No hay filas limpias para evaluar la calibración.")

        X = clean[FEATURE_COLUMNS]
        y_true = clean[TARGET_COL].values
        y_prob = self.model.predict_proba(X)[:, 1]

        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform",
        )

        report = {
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 5),
            "log_loss": round(float(log_loss(y_true, y_prob)), 5),
            "fraction_of_positives": fraction_pos,
            "mean_predicted_value": mean_pred,
            "n_samples": len(clean),
        }

        logger.info(
            "Calibración — Brier: %.4f, LogLoss: %.4f, muestras: %d",
            report["brier_score"], report["log_loss"], report["n_samples"],
        )
        return report

    # ── Feature importances ───────────────────────────────────────────────────

    def feature_importances(self) -> pd.Series:
        """
        Retorna la importancia media de cada feature en los calibradores internos.

        CalibratedClassifierCV entrena múltiples estimadores base (uno por fold).
        Se promedian sus importancias para obtener un ranking estable.

        Returns:
            ``pd.Series`` indexada por nombre de feature, ordenada de mayor a menor.

        Raises:
            RuntimeError: Si el modelo no ha sido entrenado.
        """
        if not self.is_fitted():
            raise RuntimeError("Llama a fit() antes de feature_importances().")

        importances_list: list[np.ndarray] = []
        for calibrated_clf in self.model.calibrated_classifiers_:
            base = calibrated_clf.estimator
            importances_list.append(base.feature_importances_)

        mean_imp = np.mean(importances_list, axis=0)
        series = pd.Series(mean_imp, index=FEATURE_COLUMNS, name="importance")
        return series.sort_values(ascending=False)
