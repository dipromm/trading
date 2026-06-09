"""
El Cazador — Detección de actividad inusual de insiders (SEC Form 4).

No usa ML. Son reglas condicionales sobre datos de OpenInsider.

Regla principal:
    Si un CEO/CFO/Director vende > SELL_THRESHOLD% de sus acciones
    en los últimos LOOKBACK_DAYS días → señal bajista (alerta = 1)

IMPORTANTE sobre el lag temporal:
    Las declaraciones Form 4 tienen hasta 2 días hábiles de plazo desde
    la transacción. Aplicar siempre FORM4_LAG_DAYS en el backtester.
    Esto se hace shift(FORM4_LAG_DAYS) sobre los datos de actividad.

Fuente de datos: https://openinsider.com (scraping con pausa entre requests)
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from agents.base_agent import AgentBase
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Roles de insiders considerados relevantes para la señal
RELEVANT_ROLES = {"CEO", "CFO", "Director", "President", "COO", "CTO"}


class Cazador(AgentBase):
    """
    Agente de detección de actividad de insiders basado en reglas.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        cfg = config["cazador"]
        self.sell_threshold_pct = cfg["sell_threshold_pct"]
        self.lookback_days = cfg["lookback_days"]
        self.form4_lag_days = cfg["form4_lag_days"]
        self._insider_data: Optional[pd.DataFrame] = None
        self._is_fitted = True  # No hay entrenamiento, solo reglas

    def fit(self, train_data: pd.DataFrame) -> None:
        """El Cazador no se entrena. Solo carga los datos de insiders."""
        logger.debug("El Cazador no requiere entrenamiento (basado en reglas).")
        self._is_fitted = True

    def fetch_insider_activity(self, ticker: str) -> pd.DataFrame:
        """
        Descarga el historial de transacciones de insiders para un ticker.

        Fuente: OpenInsider (scraping con pausa entre requests).

        Returns:
            DataFrame con columnas: date, role, transaction_type, shares_sold_pct
        """
        # TODO: implementar en Fase 5
        # URL de ejemplo: https://openinsider.com/screener?s=AAPL&fd=-1
        # Usar requests + BeautifulSoup para parsear la tabla de transacciones
        # Aplicar time.sleep(2) entre requests para no saturar el servidor
        raise NotImplementedError("Implementar en Fase 5")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna señal de alerta por fecha.

        La señal ya incluye el lag de FORM4_LAG_DAYS (la transacción
        del día T se reporta como señal en T + form4_lag_days).

        Returns:
            Serie binaria: 1 = alerta de insider activa, 0 = sin alerta.
        """
        # TODO: implementar en Fase 5
        # 1. Cargar datos de insiders (fetch_insider_activity o caché)
        # 2. Para cada fecha en data.index:
        #    - Buscar ventas de roles relevantes en los últimos lookback_days
        #    - Si alguna vende > sell_threshold_pct → alerta = 1
        # 3. Aplicar shift(form4_lag_days) para el lag de reporte
        # 4. Retornar pd.Series(alertas, index=data.index)
        raise NotImplementedError("Implementar en Fase 5")
