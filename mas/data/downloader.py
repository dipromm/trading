"""
Pipeline de descarga, caché y verificación de calidad de datos OHLCV.

Uso:
    python -m mas.data.downloader           # Descarga todos los tickers
    python -m mas.data.downloader --force   # Fuerza re-descarga ignorando caché

El flujo es:
    1. Leer tickers del CSV fijado a fecha de inicio
    2. Para cada ticker: usar caché Parquet si existe, si no descargar de yfinance
    3. Ejecutar quality checks (NaN, duplicados, precios negativos)
    4. Guardar en caché y retornar dict {ticker: DataFrame}
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from mas.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def load_tickers(config: dict) -> list[str]:
    """Lee el universo de tickers del CSV fijado a fecha de inicio."""
    tickers_file = Path(config["universe"]["tickers_file"])
    if not tickers_file.exists():
        raise FileNotFoundError(f"Archivo de tickers no encontrado: {tickers_file}")
    df = pd.read_csv(tickers_file)
    tickers = df["ticker"].tolist()
    logger.info(f"Universo: {len(tickers)} tickers cargados de {tickers_file.name}")
    return tickers


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path,
    force_download: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Descarga OHLCV para un ticker con caché en Parquet.

    Args:
        ticker: Símbolo del ticker (ej: "AAPL")
        start: Fecha inicio en formato "YYYY-MM-DD"
        end: Fecha fin en formato "YYYY-MM-DD"
        cache_dir: Directorio donde guardar el Parquet
        force_download: Si True, ignora el caché y descarga de nuevo

    Returns:
        DataFrame con columnas Open, High, Low, Close, Volume, o None si falla.
    """
    cache_file = cache_dir / f"{ticker}.parquet"

    if cache_file.exists() and not force_download:
        logger.debug(f"[{ticker}] Desde caché")
        return pd.read_parquet(cache_file)

    logger.info(f"[{ticker}] Descargando ({start} -> {end})")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        if df.empty:
            logger.warning(f"[{ticker}] yfinance devolvió DataFrame vacío")
            return None

        # yfinance puede devolver MultiIndex al descargar un solo ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.debug(f"[{ticker}] {len(df)} filas guardadas en caché")
        return df

    except Exception as exc:
        logger.error(f"[{ticker}] Error en descarga: {exc}")
        return None


def quality_check(df: pd.DataFrame, ticker: str) -> bool:
    """
    Verifica la integridad de los datos OHLCV.

    Checks:
        - Sin NaN en columnas OHLCV
        - Sin fechas duplicadas
        - Precios de cierre estrictamente positivos
        - Avisa si hay gaps > 7 días calendario (puede ser normal en festivos)

    Returns:
        True si todos los checks críticos pasan.
    """
    passed = True

    nan_counts = df[["Open", "High", "Low", "Close", "Volume"]].isna().sum()
    if nan_counts.any():
        logger.warning(f"[{ticker}] NaN encontrados: {nan_counts[nan_counts > 0].to_dict()}")
        passed = False

    if df.index.duplicated().any():
        n = df.index.duplicated().sum()
        logger.warning(f"[{ticker}] {n} fechas duplicadas")
        passed = False

    if (df["Close"] <= 0).any():
        n = (df["Close"] <= 0).sum()
        logger.warning(f"[{ticker}] {n} precios de cierre negativos o cero")
        passed = False

    # Gap check (informativo, no falla el check)
    if len(df) > 1:
        gaps = df.index.to_series().diff().dt.days.dropna()
        max_gap = gaps.max()
        if max_gap > 7:
            logger.warning(f"[{ticker}] Gap máximo entre fechas: {max_gap} días. Verifica si es correcto.")

    return passed


def download_all(
    config: Optional[dict] = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Descarga todos los tickers del universo y retorna datos limpios.

    Args:
        config: Configuración del proyecto. Si None, carga config.yaml.
        force_download: Si True, re-descarga ignorando caché.

    Returns:
        dict {ticker: DataFrame OHLCV} para los tickers que pasaron la descarga.
        Los tickers que fallaron se registran en el log pero no se incluyen.
    """
    if config is None:
        config = load_config()

    tickers = load_tickers(config)
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    cache_dir = Path(config["data"]["cache_dir"])

    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tqdm(tickers, desc="Descargando tickers"):
        df = download_ticker(ticker, start, end, cache_dir, force_download)

        if df is None:
            failed.append(ticker)
            continue

        quality_check(df, ticker)
        data[ticker] = df

        # Pausa solo si se acaba de descargar (no está en caché)
        cache_file = cache_dir / f"{ticker}.parquet"
        if not cache_file.exists():
            time.sleep(0.3)

    if failed:
        logger.error(f"Tickers fallidos ({len(failed)}): {failed}")

    logger.info(f"Pipeline completado: {len(data)}/{len(tickers)} tickers OK")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga datos OHLCV del universo de tickers.")
    parser.add_argument("--force", action="store_true", help="Fuerza re-descarga ignorando caché")
    args = parser.parse_args()

    from mas.utils.reproducibility import set_all_seeds
    cfg = load_config()
    set_all_seeds(cfg["general"]["random_seed"])

    data = download_all(config=cfg, force_download=args.force)
    print(f"\nResumen: {len(data)} tickers descargados correctamente.")
    for ticker, df in list(data.items())[:3]:
        print(f"  {ticker}: {len(df)} filas | {df.index[0].date()} -> {df.index[-1].date()}")
    if len(data) > 3:
        print(f"  ... y {len(data) - 3} más.")
