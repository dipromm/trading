"""
El Analista — Sentimiento de noticias financieras con FinBERT.

Pipeline:
    1. Recibe titulares de noticias por ticker y fecha
    2. Procesa cada titular con FinBERT (positivo/negativo/neutral)
    3. Agrega los scores del día en una probabilidad media
    4. Calibra con Platt Scaling
    5. Retorna probabilidad calibrada p ∈ [0, 1] al Juez

IMPORTANTE sobre datos históricos:
    Las noticias de fin de semana se asignan al lunes siguiente.
    El backtester solo opera en días hábiles (NYSE calendar).
    Noticias publicadas después del cierre del mercado se asignan al día siguiente.

Fuente de datos recomendada: Alpaca News API (gratuita para histórico).
"""

import logging
from pathlib import Path

import pandas as pd

from agents.base_agent import AgentBase
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


class Analista(AgentBase):
    """
    Agente de análisis de sentimiento basado en FinBERT.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config["analista"]
        self._pipeline = None   # Se carga lazy para no consumir RAM hasta que se use
        self._calibrator = None
        self._is_fitted = False

    def _load_finbert(self) -> None:
        """Carga el pipeline de FinBERT desde Hugging Face (lazy load)."""
        # TODO: implementar en Fase 4
        # from transformers import pipeline
        # self._pipeline = pipeline(
        #     "sentiment-analysis",
        #     model=self.config["model_name"],
        #     device=0 if torch.cuda.is_available() else -1,
        # )
        raise NotImplementedError("Implementar en Fase 4")

    def score_headline(self, headline: str) -> float:
        """
        Puntúa un titular con FinBERT.

        Returns:
            Score ∈ [0, 1]: 1 = muy positivo, 0 = muy negativo, 0.5 = neutral.
        """
        # TODO: implementar en Fase 4
        # result = self._pipeline(headline[:self.config["max_length"]])[0]
        # if result["label"] == "positive": return 0.5 + result["score"] / 2
        # if result["label"] == "negative": return 0.5 - result["score"] / 2
        # return 0.5  # neutral
        raise NotImplementedError("Implementar en Fase 4")

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entrena el calibrador de probabilidades sobre datos históricos.

        Args:
            train_data: DataFrame con columnas 'sentiment_raw' (score FinBERT
                        agregado por día) y 'target_binary'.
        """
        # TODO: implementar en Fase 4
        # 1. Calibrar self._calibrator con Platt Scaling sobre sentiment_raw vs target
        # 2. self._is_fitted = True
        raise NotImplementedError("Implementar en Fase 4")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna probabilidad calibrada de sentimiento positivo por fecha.

        Returns:
            Serie de probabilidades p ∈ [0, 1] por fecha.
        """
        # TODO: implementar en Fase 4
        raise NotImplementedError("Implementar en Fase 4")
