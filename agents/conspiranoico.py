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

Tecnología: Isolation Forest (scikit-learn) + HMM opcional (hmmlearn)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from agents.base_agent import AgentBase
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


class Conspiranoico(AgentBase):
    """
    Agente de detección de régimen de mercado anómalo.
    Emite veto cuando detecta condiciones de alta incertidumbre.
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
        self.volatility_windows = cfg["volatility_windows"]
        self.correlation_window = cfg["correlation_window"]
        self.volume_ratio_window = cfg["volume_ratio_window"]
        self._veto_threshold: Optional[float] = None
        self._is_fitted = False

    def build_regime_features(
        self,
        returns: pd.DataFrame,
        vix: pd.Series,
        volumes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construye las features de régimen de mercado.

        Args:
            returns: DataFrame {ticker: retorno_diario} para todo el universo
            vix: Serie del VIX diario (descargada como ^VIX de yfinance)
            volumes: DataFrame {ticker: volume_ratio} para todo el universo

        Returns:
            DataFrame con features de régimen por fecha.
        """
        # TODO: implementar en Fase 6
        # Features a calcular:
        # 1. Para cada window en volatility_windows:
        #    vol_{w}d = retornos.std(axis=1).rolling(w).mean()  ← volatilidad media del universo
        # 2. vix = vix (normalizado)
        # 3. avg_correlation = correlación media rolling entre todos los tickers
        # 4. avg_volume_ratio = media del ratio de volumen del universo
        # 5. market_breadth = % de tickers con retorno > 0 ese día
        raise NotImplementedError("Implementar en Fase 6")

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entrena el Isolation Forest sobre las features de régimen del período de entrenamiento.

        Args:
            train_data: DataFrame con las features devueltas por build_regime_features().
        """
        # TODO: implementar en Fase 6
        # 1. Eliminar NaN
        # 2. self.model.fit(X)  ← Isolation Forest no supervisa, no necesita target
        # 3. Calcular el percentil 95 del anomaly score en train como threshold
        # 4. self._is_fitted = True
        raise NotImplementedError("Implementar en Fase 6")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna señal de veto por fecha.

        Returns:
            Serie binaria: 1 = veto activo (mercado anómalo), 0 = mercado normal.
        """
        # TODO: implementar en Fase 6
        # 1. scores = self.model.score_samples(X)  ← más negativo = más anómalo
        # 2. veto = (scores < self._veto_threshold).astype(int)
        # 3. Retornar pd.Series(veto, index=data.index)
        raise NotImplementedError("Implementar en Fase 6")
