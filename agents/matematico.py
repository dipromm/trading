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

import logging

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from agents.base_agent import AgentBase
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Features que usa este agente (deben existir en el DataFrame de entrada)
FEATURE_COLUMNS = [
    "close_norm", "return_1d", "return_5d", "return_20d",
    "rsi_14", "rsi_28",
    "macd", "macd_signal", "macd_hist",
    "bb_bandwidth", "bb_pct_b",
    "atr_pct",
    "volume_ratio",
    "sma_20_50_ratio", "sma_50_200_ratio",
]


class Matematico(AgentBase):
    """
    Agente de análisis técnico basado en XGBoost con probabilidades calibradas.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        cfg = config["matematico"]

        base_model = XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            random_state=cfg["random_state"],
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self.model = CalibratedClassifierCV(
            estimator=base_model,
            method=cfg["calibration_method"],  # "sigmoid" = Platt Scaling
            cv=5,
        )
        self._is_fitted = False

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entrena el clasificador binario y calibra las probabilidades.

        Args:
            train_data: DataFrame con FEATURE_COLUMNS + 'target_binary'.
                        Solo datos del período de entrenamiento walk-forward.
        """
        # TODO: implementar
        # 1. Eliminar filas con NaN en features o target
        # 2. Separar X (features) e y (target_binary)
        # 3. self.model.fit(X, y)  ← CalibratedClassifierCV hace calibración interna
        # 4. self._is_fitted = True
        # 5. Loggear: n filas entrenadas, distribución de clases, etc.
        raise NotImplementedError("Implementar en Fase 3")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna la probabilidad calibrada de subida para cada día.

        Returns:
            Serie de probabilidades p ∈ [0, 1] por fecha.
            p > 0.5 → señal alcista; p < 0.5 → señal bajista.
        """
        # TODO: implementar
        # 1. Verificar self.is_fitted()
        # 2. Extraer FEATURE_COLUMNS del DataFrame
        # 3. proba = self.model.predict_proba(X)[:, 1]  ← probabilidad de clase 1 (sube)
        # 4. Retornar pd.Series(proba, index=data.index)
        raise NotImplementedError("Implementar en Fase 3")
