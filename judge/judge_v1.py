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
"""

import logging
from typing import Optional

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from utils.config_loader import load_config

logger = logging.getLogger(__name__)


class JuezV1:
    """
    Meta-modelo que combina las señales de los agentes.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        cfg = config["judge_v1"]

        if cfg["meta_model"] == "logistic":
            base = LogisticRegression(random_state=cfg["random_state"], max_iter=1000)
        else:
            base = XGBClassifier(random_state=cfg["random_state"], eval_metric="logloss")

        self.model = CalibratedClassifierCV(
            estimator=base,
            method=cfg["calibration_method"],
            cv=5,
        )
        self._is_pass_through = True  # Iter 1: pasa la señal del Matemático directamente
        self._is_fitted = False

    def fit(
        self,
        agent_predictions: pd.DataFrame,
        targets: pd.Series,
    ) -> None:
        """
        Entrena el meta-modelo con las predicciones de los agentes.

        Args:
            agent_predictions: DataFrame donde cada columna es la señal
                               de un agente (matemático, analista, etc.)
                               Estas predicciones deben ser de un período ANTERIOR
                               al de validación actual (sin data leakage).
            targets: Serie binaria con el resultado real (subió/bajó)
        """
        # TODO: implementar en Fase 4
        # 1. Eliminar NaN
        # 2. self.model.fit(agent_predictions, targets)
        # 3. self._is_pass_through = False
        # 4. self._is_fitted = True
        raise NotImplementedError("Implementar en Fase 4. En Fase 3 usar predict() en modo pass-through.")

    def predict(
        self,
        agent_predictions: pd.DataFrame,
        veto_signal: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Emite la decisión final por ticker y fecha.

        Args:
            agent_predictions: DataFrame {ticker: señal del agente} por fecha.
                               En Fase 3, solo contiene la señal del Matemático.
            veto_signal: Serie del Conspiranoico (1=veto, 0=normal).
                        Si None, se ignora el veto.

        Returns:
            DataFrame {ticker: probabilidad final} ∈ [0, 1].
            Días con veto activo tienen valor 0 para todos los tickers.
        """
        # TODO: implementar en Fase 3 (pass-through) y Fase 4 (meta-modelo)
        # Fase 3 (pass-through):
        #   result = agent_predictions.copy()
        #
        # Fase 4+ (meta-modelo):
        #   result = pd.DataFrame(self.model.predict_proba(agent_predictions)[:,1], ...)
        #
        # Aplicar veto en ambos casos:
        #   if veto_signal is not None:
        #       result.loc[veto_signal == 1, :] = 0.0
        raise NotImplementedError("Implementar en Fase 3")
