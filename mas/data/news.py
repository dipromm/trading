"""
Pipeline de descarga, cache y procesamiento de noticias financieras.

Fuente principal: Alpaca News API (gratuita con cuenta paper trading).
Requiere variables de entorno ALPACA_API_KEY y ALPACA_SECRET_KEY.

Flujo:
    1. Descargar titulares de noticias por ticker via Alpaca Data API v2
    2. Cachear en Parquet por ticker (data/cache/news/)
    3. Asignar cada noticia al dia de mercado correspondiente (NYSE calendar)
    4. Retornar {ticker: DataFrame} con columnas [date, headline, created_at, source]

Sincronizacion temporal:
    - Noticias publicadas durante el horario de mercado -> ese dia
    - Noticias publicadas despues del cierre -> siguiente dia habil
    - Noticias de fin de semana / festivos -> siguiente dia habil
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mas.utils.config_loader import load_config

logger = logging.getLogger(__name__)

_ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
_PAGE_LIMIT = 50
_REQUEST_DELAY_S = 0.35  # ~170 req/min, below 200/min free tier
_MAX_RETRIES = 3


def _get_alpaca_credentials() -> tuple[str, str] | None:
    """Returns (api_key, secret_key) or None if not configured."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return None
    return api_key, secret_key


def _fetch_page(
    ticker: str,
    start: str,
    end: str,
    page_token: str | None,
    api_key: str,
    secret_key: str,
) -> tuple[list[dict], str | None]:
    """Fetch a single page from the Alpaca News API."""
    import requests

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    params: dict = {
        "symbols": ticker,
        "start": start,
        "end": end,
        "limit": _PAGE_LIMIT,
        "sort": "asc",
    }
    if page_token:
        params["page_token"] = page_token

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                _ALPACA_NEWS_URL,
                headers=headers,
                params=params,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited, esperando %ds...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("news", []), data.get("next_page_token")
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                logger.error("Error descargando noticias de %s: %s", ticker, exc)
                return [], None
            time.sleep(1)

    return [], None


def download_news_ticker(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    secret_key: str,
) -> pd.DataFrame:
    """
    Download all news for a single ticker via Alpaca API (paginated).

    Returns DataFrame with columns: [headline, created_at, source, symbols].
    """
    all_articles: list[dict] = []
    page_token: str | None = None

    while True:
        articles, page_token = _fetch_page(
            ticker, start, end, page_token, api_key, secret_key,
        )
        if not articles:
            break
        all_articles.extend(articles)
        if page_token is None:
            break
        time.sleep(_REQUEST_DELAY_S)

    if not all_articles:
        return pd.DataFrame(columns=["headline", "created_at", "source", "symbols"])

    records = []
    for art in all_articles:
        records.append({
            "headline": art.get("headline", ""),
            "created_at": art.get("created_at", ""),
            "source": art.get("source", ""),
            "symbols": ",".join(art.get("symbols", [])),
        })

    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def download_all_news(
    tickers: list[str],
    config: dict | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download and cache news for all tickers.

    Uses Parquet cache in data/cache/news/ to avoid re-downloading.

    Returns:
        {ticker: DataFrame} with raw news articles.
        Empty dict if API credentials are not configured.
    """
    if config is None:
        config = load_config()

    creds = _get_alpaca_credentials()
    if creds is None:
        logger.warning(
            "Variables ALPACA_API_KEY / ALPACA_SECRET_KEY no configuradas. "
            "No se pueden descargar noticias. El Analista se omitira."
        )
        return {}

    api_key, secret_key = creds
    cache_dir = Path(config["data"]["cache_dir"]) / "news"
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    result: dict[str, pd.DataFrame] = {}
    download_needed = []

    for ticker in tickers:
        cache_file = cache_dir / f"{ticker}.parquet"
        if cache_file.exists() and not force_download:
            try:
                df = pd.read_parquet(cache_file)
                result[ticker] = df
                continue
            except Exception:
                pass
        download_needed.append(ticker)

    if download_needed:
        logger.info(
            "Descargando noticias de Alpaca para %d tickers...",
            len(download_needed),
        )
        for ticker in tqdm(download_needed, desc="Noticias", unit="ticker"):
            df = download_news_ticker(ticker, start, end, api_key, secret_key)
            cache_file = cache_dir / f"{ticker}.parquet"
            if not df.empty:
                df.to_parquet(cache_file, index=False)
                logger.debug("%s: %d articulos descargados", ticker, len(df))
            else:
                df.to_parquet(cache_file, index=False)
                logger.debug("%s: sin noticias", ticker)
            result[ticker] = df
    else:
        logger.info("Noticias cargadas de cache para %d tickers", len(result))

    total = sum(len(df) for df in result.values())
    logger.info("Total: %d articulos para %d tickers", total, len(result))
    return result


def load_cached_news(config: dict | None = None) -> dict[str, pd.DataFrame]:
    """
    Load all cached news from Parquet without downloading.

    Returns:
        {ticker: DataFrame} with cached news articles.
        Empty dict if no cache exists.
    """
    if config is None:
        config = load_config()

    cache_dir = Path(config["data"]["cache_dir"]) / "news"
    if not cache_dir.exists():
        return {}

    result: dict[str, pd.DataFrame] = {}
    for parquet_file in cache_dir.glob("*.parquet"):
        ticker = parquet_file.stem
        try:
            df = pd.read_parquet(parquet_file)
            if not df.empty:
                result[ticker] = df
        except Exception as exc:
            logger.warning("Error leyendo cache de %s: %s", ticker, exc)

    return result


def assign_to_trading_days(
    news_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Assign each news article to its effective trading day.

    Rules (from docs/PLAN.md):
        - During market hours: assigned to that day
        - After market close / weekends / holidays: next trading day
        - NYSE closes at 16:00 ET

    Args:
        news_df: DataFrame with 'created_at' column (UTC timestamps).
        trading_dates: DatetimeIndex of NYSE trading days.

    Returns:
        Same DataFrame with 'trading_day' column added.
    """
    if news_df.empty or trading_dates.empty:
        news_df["trading_day"] = pd.NaT
        return news_df

    created = news_df["created_at"]
    if created.dt.tz is None:
        created = created.dt.tz_localize("UTC")

    eastern = created.dt.tz_convert("US/Eastern")

    article_dates = eastern.dt.normalize().dt.tz_localize(None)
    article_hours = eastern.dt.hour

    # NYSE closes at 16:00 ET; articles after close go to next trading day
    is_after_close = article_hours >= 16

    effective_dates = article_dates.copy()
    effective_dates[is_after_close] += pd.Timedelta(days=1)

    td_sorted = trading_dates.sort_values()

    trading_days = []
    for edate in effective_dates:
        idx = td_sorted.searchsorted(edate, side="left")
        if idx < len(td_sorted):
            trading_days.append(td_sorted[idx])
        else:
            trading_days.append(pd.NaT)

    result = news_df.copy()
    result["trading_day"] = trading_days
    return result


def get_trading_dates(
    start: str,
    end: str,
) -> pd.DatetimeIndex:
    """Get NYSE trading dates for the given period."""
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return schedule.index
