"""
El Cazador — Detección de actividad inusual de insiders (SEC Form 4).

No usa ML. Son reglas condicionales sobre transacciones Form 4 de SEC EDGAR.

Regla principal:
    Si un CEO/CFO/Director vende > SELL_THRESHOLD% de sus acciones
    en los últimos LOOKBACK_DAYS días → señal bajista (alerta = 1)

    IMPORTANTE sobre el lag temporal:
    Las declaraciones Form 4 tienen hasta 2 días hábiles de plazo desde
    la transacción. Aplicar siempre FORM4_LAG_DAYS en el backtester.
    Esto se hace usando filing_date + shift(FORM4_LAG_DAYS) sobre días hábiles.

Fuente de datos: SEC EDGAR (data.sec.gov), ver mas/data/insiders.py
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from mas.agents.base_agent import AgentBase
from mas.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Roles de insiders considerados relevantes para la señal
RELEVANT_ROLES = {"CEO", "CFO", "Director", "President", "COO", "CTO"}


class Cazador(AgentBase):
    """
    Agente de detección de actividad de insiders basado en reglas.

    No requiere entrenamiento. Las alertas se calculan a partir de los
    datos de SEC EDGAR cacheados y las reglas configuradas.

    Flujo de uso:
        1. (Una vez antes del walk-forward) Llamar a precompute_insider_signals()
           para descargar y cachear los datos de SEC EDGAR.
        2. En cada ventana walk-forward, fit() es no-op.
        3. predict(val_data) retorna una Serie binaria 0/1 por fecha.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        self._full_config = config
        cfg = config["cazador"]
        self.sell_threshold_pct: float = cfg["sell_threshold_pct"]
        self.lookback_days: int = cfg["lookback_days"]
        self.form4_lag_days: int = cfg["form4_lag_days"]
        self._insider_data: dict[str, pd.DataFrame] = {}
        self._is_fitted = True  # No hay entrenamiento, solo reglas

    def fit(self, train_data: pd.DataFrame) -> None:
        """El Cazador no se entrena. No-op requerido por AgentBase."""
        logger.debug("El Cazador no requiere entrenamiento (basado en reglas).")
        self._is_fitted = True

    def precompute_insider_signals(
        self,
        tickers: list[str],
        force_download: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Descarga y cachea los datos de insiders para todos los tickers equity.

        Equivalente a Analista.precompute_sentiment(): se llama una vez antes
        del walk-forward para evitar repetir el scraping en cada ventana.

        Args:
            tickers: Lista completa de tickers del universo.
            force_download: Si True, re-descarga ignorando el caché.

        Returns:
            {ticker: DataFrame} con transacciones de insiders cacheadas.
        """
        from mas.data.insiders import download_all_insiders

        self._insider_data = download_all_insiders(
            tickers=tickers,
            config=self._full_config,
            force_download=force_download,
        )
        logger.info(
            "Cazador: datos de insiders cargados para %d tickers equity",
            len(self._insider_data),
        )
        return self._insider_data

    def fetch_insider_activity(self, ticker: str) -> pd.DataFrame:
        """
        Retorna el historial de transacciones de insiders para un ticker.

        Si no hay datos precargados, intenta cargar desde caché o descarga.

        Returns:
            DataFrame con columnas: trade_date, filing_date, role,
            transaction_type, shares_sold_pct.
        """
        if ticker in self._insider_data:
            return self._insider_data[ticker]

        from mas.data.insiders import download_insider_ticker, load_cached_insiders

        # Intentar cargar desde caché primero
        cached = load_cached_insiders([ticker], self._full_config)
        if ticker in cached:
            self._insider_data[ticker] = cached[ticker]
            return cached[ticker]

        # Descargar si no hay caché
        logger.info("Cazador: descargando insiders para %s...", ticker)
        cfg_data = self._full_config["data"]
        df = download_insider_ticker(
            ticker,
            start_date=cfg_data["start_date"],
            end_date=cfg_data["end_date"],
        )
        self._insider_data[ticker] = df
        return df

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Retorna señal de alerta por fecha.

        Para el uso dentro del walk-forward, donde se conoce el ticker,
        es preferible llamar a predict_ticker(data, ticker) directamente.

        Args:
            data: DataFrame indexado por fechas. Se usa solo el índice.

        Returns:
            pd.Series binaria (dtype=int): 1 = alerta, 0 = silencio.
        """
        # Silencio si no hay datos de insiders cargados
        from mas.data.insiders import build_alert_series

        trading_dates = data.index
        if not isinstance(trading_dates, pd.DatetimeIndex):
            trading_dates = pd.DatetimeIndex(trading_dates)

        return build_alert_series(
            insider_df=pd.DataFrame(),
            trading_dates=trading_dates,
            sell_threshold_pct=self.sell_threshold_pct,
            lookback_days=self.lookback_days,
            form4_lag_days=self.form4_lag_days,
        )

    def predict_ticker(self, data: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Retorna señal de alerta por fecha para un ticker específico.

        Método preferido en el walk-forward donde el ticker se conoce
        como clave externa del diccionario val_feats.

        Args:
            data: DataFrame de features del período de validación,
                  indexado por fechas de trading. Se usa solo el índice.
            ticker: Símbolo del activo (ej. "AAPL").

        Returns:
            pd.Series binaria (dtype=int): 1 = alerta de insider, 0 = silencio.
            Indexada con las mismas fechas que data.index.
        """
        from mas.data.insiders import build_alert_series

        insider_df = self.fetch_insider_activity(ticker)

        trading_dates = data.index
        if not isinstance(trading_dates, pd.DatetimeIndex):
            trading_dates = pd.DatetimeIndex(trading_dates)

        return build_alert_series(
            insider_df=insider_df,
            trading_dates=trading_dates,
            sell_threshold_pct=self.sell_threshold_pct,
            lookback_days=self.lookback_days,
            form4_lag_days=self.form4_lag_days,
            ticker=ticker,
        )
