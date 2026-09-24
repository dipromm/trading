"""
Tests de integridad de datos.

Verifican que el pipeline de datos no introduce NaN, gaps ni fechas duplicadas
que podrían afectar silenciosamente a los modelos.

Ejecutar después de cada descarga de datos (data/downloader.py).
"""

import pytest
import pandas as pd
from pathlib import Path


def load_cached_tickers() -> list[str]:
    """Retorna los tickers disponibles en caché."""
    cache_dir = Path("data/cache")
    if not cache_dir.exists():
        return []
    return [f.stem for f in cache_dir.glob("*.parquet")]


@pytest.mark.skipif(
    not Path("data/cache").exists(),
    reason="Caché de datos no disponible. Ejecutar python -m mas.data.downloader primero."
)
class TestDataIntegrity:

    @pytest.fixture(params=load_cached_tickers())
    def ticker_data(self, request):
        ticker = request.param
        df = pd.read_parquet(f"data/cache/{ticker}.parquet")
        return ticker, df

    def test_no_nan_in_close(self, ticker_data):
        """Sin NaN en el precio de cierre."""
        ticker, df = ticker_data
        nan_count = df["Close"].isna().sum()
        assert nan_count == 0, f"[{ticker}] {nan_count} NaN en Close"

    def test_no_duplicate_dates(self, ticker_data):
        """Sin fechas duplicadas en el índice."""
        ticker, df = ticker_data
        n_dups = df.index.duplicated().sum()
        assert n_dups == 0, f"[{ticker}] {n_dups} fechas duplicadas"

    def test_close_prices_positive(self, ticker_data):
        """Todos los precios de cierre deben ser positivos."""
        ticker, df = ticker_data
        n_invalid = (df["Close"] <= 0).sum()
        assert n_invalid == 0, f"[{ticker}] {n_invalid} precios de cierre <= 0"

    def test_date_index_is_monotonic(self, ticker_data):
        """El índice de fechas debe ser estrictamente creciente."""
        ticker, df = ticker_data
        assert df.index.is_monotonic_increasing, \
            f"[{ticker}] El índice de fechas no es monotónicamente creciente"

    def test_minimum_rows(self, ticker_data):
        """Cada ticker debe tener al menos 1000 filas (≈4 años de datos diarios)."""
        ticker, df = ticker_data
        assert len(df) >= 1000, \
            f"[{ticker}] Solo {len(df)} filas. Se esperaban al menos 1000."
