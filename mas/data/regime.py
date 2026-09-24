"""
Features de régimen de mercado para El Conspiranoico.

Calcula señales a nivel de mercado (no por ticker) que sirven como
inputs del Isolation Forest para detectar regímenes anómalos.

Todas las features son lookback-safe: el valor del día T solo usa
datos hasta T (sin shift(-1) ni acceso a datos futuros).

Uso:
    from mas.data.regime import download_vix, build_regime_features
    vix = download_vix(config)
    regime_df = build_regime_features(prices, vix, config)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mas.utils.config_loader import load_config

logger = logging.getLogger(__name__)

_VIX_DOWNLOAD_RETRIES = 3
_VIX_RETRY_SLEEP_SEC = 2.0
_DEFAULT_VIX_FALLBACKS = ("^VIX", "VIXY")


def _normalize_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza salida de yfinance (MultiIndex, tz, columnas)."""
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def _slice_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Recorta a [start, end] inclusive."""
    if df.empty:
        return df
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def _fetch_vix_from_yfinance(ticker_sym: str, start: str, end: str) -> pd.DataFrame:
    """
    Intenta varias APIs de yfinance; Yahoo a veces falla con download() pero
    responde con Ticker.history() o period='max'.
    """
    import yfinance as yf

    # yfinance trata ``end`` como exclusivo en muchas versiones
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            ticker_sym,
            start=start,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        df = _normalize_yfinance_df(raw)
        if not df.empty and "Close" in df.columns:
            return _slice_date_range(df, start, end)
    except Exception as exc:
        logger.debug("[VIX] yf.download(%s) error: %s", ticker_sym, exc)

    try:
        raw = yf.Ticker(ticker_sym).history(
            start=start,
            end=end_exclusive,
            auto_adjust=True,
        )
        df = _normalize_yfinance_df(raw)
        if not df.empty and "Close" in df.columns:
            return _slice_date_range(df, start, end)
    except Exception as exc:
        logger.debug("[VIX] Ticker.history(%s) error: %s", ticker_sym, exc)

    try:
        raw = yf.Ticker(ticker_sym).history(period="max", auto_adjust=True)
        df = _normalize_yfinance_df(raw)
        if not df.empty and "Close" in df.columns:
            return _slice_date_range(df, start, end)
    except Exception as exc:
        logger.debug("[VIX] Ticker.history(period=max, %s) error: %s", ticker_sym, exc)

    return pd.DataFrame()


def _vix_ticker_candidates(config: dict) -> list[str]:
    primary = config["conspiranoico"]["vix_ticker"]
    extra = config["conspiranoico"].get("vix_fallback_tickers", _DEFAULT_VIX_FALLBACKS)
    seen: set[str] = set()
    ordered: list[str] = []
    for sym in (primary, *extra):
        if sym and sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    return ordered


def download_vix(
    config: Optional[dict] = None,
    force_download: bool = False,
) -> pd.Series:
    """
    Descarga el VIX (^VIX) de yfinance con caché Parquet.

    El VIX se descarga con el mismo rango de fechas que el universo principal
    y se cachea en ``data/cache/vix_index.parquet``.

    Estrategia ante fallos de Yahoo:
        1. ``yf.download`` → ``Ticker.history`` → ``history(period='max')``
        2. Tickers alternativos (p. ej. VIXY como proxy de liquidez)
        3. Reintentos con pausa
        4. Caché local obsoleta si la descarga sigue fallando

    Returns:
        Serie de VIX diario indexada por fecha (DatetimeIndex).
        En fechas donde no hay dato (festivos NYSE), el valor se rellena
        con forward-fill para mantener el índice continuo.
    """
    if config is None:
        config = load_config()

    cache_dir = Path(config["data"]["cache_dir"])
    cache_file = cache_dir / "vix_index.parquet"
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    target_end = pd.Timestamp(end).normalize()
    tickers = _vix_ticker_candidates(config)

    if cache_file.exists() and not force_download:
        cached_df = pd.read_parquet(cache_file)
        cached_df.index = pd.to_datetime(cached_df.index).normalize()
        cache_end = cached_df.index.max()
        if cache_end >= target_end - pd.Timedelta(days=4):
            logger.debug("[VIX] Cargando desde caché (hasta %s)", cache_end.date())
            return cached_df["Close"]

        inc_start = (cache_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fetch_end = (target_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("[VIX] Ampliando caché (%s -> %s)", inc_start, end)
        for ticker_sym in tickers:
            new_df = _fetch_vix_from_yfinance(ticker_sym, inc_start, fetch_end)
            if new_df.empty or "Close" not in new_df.columns:
                continue
            merged = pd.concat([cached_df, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.to_parquet(cache_file)
            logger.info("[VIX] Caché ampliada hasta %s", merged.index.max().date())
            return merged["Close"]
        logger.warning("[VIX] Sin datos nuevos; usando caché hasta %s", cache_end.date())
        return cached_df["Close"]

    logger.info("[VIX] Descargando (%s -> %s) — candidatos: %s", start, end, tickers)

    last_error: str | None = None
    for attempt in range(1, _VIX_DOWNLOAD_RETRIES + 1):
        for ticker_sym in tickers:
            df = _fetch_vix_from_yfinance(ticker_sym, start, end)
            if df.empty or "Close" not in df.columns:
                last_error = f"{ticker_sym}: sin datos"
                logger.warning("[VIX] Intento %d/%d — %s", attempt, _VIX_DOWNLOAD_RETRIES, last_error)
                continue

            if ticker_sym != tickers[0]:
                logger.warning(
                    "[VIX] Usando ticker alternativo %s (falló %s)",
                    ticker_sym, tickers[0],
                )

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file)
            logger.info("[VIX] %d filas guardadas en caché (%s)", len(df), ticker_sym)
            return df["Close"]

        if attempt < _VIX_DOWNLOAD_RETRIES:
            time.sleep(_VIX_RETRY_SLEEP_SEC)

    if cache_file.exists():
        logger.warning(
            "[VIX] Descarga fallida (%s). Usando caché local existente (puede estar desactualizada).",
            last_error,
        )
        return pd.read_parquet(cache_file)["Close"]

    raise RuntimeError(
        f"No se pudo descargar el VIX ({', '.join(tickers)}). "
        f"Último error: {last_error}. "
        "Prueba: pip install -U yfinance, reintenta más tarde, o genera "
        f"manualmente {cache_file}."
    )


def _load_equity_tickers(config: dict) -> list[str]:
    """Retorna los tickers con asset_class == 'equity' del CSV del universo."""
    try:
        tickers_file = Path(config["universe"]["tickers_file"])
        df = pd.read_csv(tickers_file)
        equity = df.loc[df["asset_class"] == "equity", "ticker"].tolist()
        logger.debug("Tickers equity para régimen: %d", len(equity))
        return equity
    except Exception as exc:
        logger.warning("No se pudo leer el CSV del universo: %s. Usando todos los tickers.", exc)
        return []


def build_returns_matrix(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye la matriz de retornos diarios Close por ticker.

    Args:
        prices: dict {ticker: OHLCV DataFrame} de download_all().

    Returns:
        DataFrame {ticker: retorno_1d} con DatetimeIndex común.
    """
    closes = {ticker: df["Close"] for ticker, df in prices.items()}
    close_df = pd.DataFrame(closes)
    return close_df.pct_change(1)


def compute_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Correlación media pairwise entre tickers de forma rolling.

    Una correlación alta (cercana a 1) indica movimiento en manada,
    señal clásica de pánico de mercado.

    El cálculo usa la correlación de la ventana rolling de retornos.
    Para eficiencia, se calcula la correlación full sobre la ventana
    usando pandas rolling.corr y se promedia el triángulo superior.

    Args:
        returns: DataFrame de retornos diarios (filas = fechas, cols = tickers).
        window: Ventana rolling en días de mercado.

    Returns:
        Serie de correlación media rolling por fecha.
    """
    if returns.shape[1] < 2:
        return pd.Series(np.nan, index=returns.index, name="avg_correlation")

    # pandas rolling.corr devuelve MultiIndex (date, ticker1) × ticker2
    # Para cada fecha, promediamos el triángulo superior de la matriz de correlación
    rolling_corr = returns.rolling(window=window, min_periods=max(window // 2, 5)).corr()

    avg_corr = {}
    dates = returns.index[window - 1:]  # primeras fechas no tienen ventana completa

    for date in returns.index:
        try:
            corr_matrix = rolling_corr.loc[date]
            if corr_matrix.isna().all().all():
                avg_corr[date] = np.nan
                continue
            # Triángulo superior sin diagonal
            n = corr_matrix.shape[0]
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            upper_vals = corr_matrix.values[mask]
            finite = upper_vals[np.isfinite(upper_vals)]
            avg_corr[date] = float(finite.mean()) if len(finite) > 0 else np.nan
        except (KeyError, TypeError):
            avg_corr[date] = np.nan

    return pd.Series(avg_corr, name="avg_correlation")


def compute_market_breadth(returns: pd.DataFrame) -> pd.Series:
    """
    Porcentaje de acciones del universo equity que suben ese día.

    Un porcentaje muy bajo (< 20%) indica capitulación generalizada;
    muy alto (> 80%) puede indicar euforia. Ambos extremos son señales
    de régimen inusual.

    Args:
        returns: DataFrame de retornos diarios (solo tickers equity).

    Returns:
        Serie [0, 1] por fecha.
    """
    positive = (returns > 0).sum(axis=1)
    total = returns.notna().sum(axis=1)
    breadth = positive / total.replace(0, np.nan)
    return breadth.rename("market_breadth")


def build_regime_features(
    prices: dict[str, pd.DataFrame],
    vix: pd.Series,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Ensambla el DataFrame de features de régimen de mercado.

    Features calculadas (todas lookback-safe):
        - vol_5d, vol_20d, vol_60d: std rolling de retornos, promediada
          sobre acciones equity del universo
        - vix: nivel VIX alineado por fecha
        - avg_correlation: correlación media pairwise rolling (equity only)
        - avg_volume_ratio: media del ratio volumen/MA20 del universo equity
        - market_breadth: % acciones equity con retorno > 0 ese día

    Args:
        prices: dict {ticker: OHLCV DataFrame} completo (equity + ETFs).
        vix: Serie de VIX diario (de download_vix()).
        config: Configuración del proyecto.

    Returns:
        DataFrame con features de régimen indexado por fecha.
        Las primeras filas tienen NaN por los lookback periods — se eliminan
        en Conspiranoico.fit().
    """
    if config is None:
        config = load_config()

    cfg = config["conspiranoico"]
    vol_windows = cfg["volatility_windows"]
    corr_window = cfg["correlation_window"]
    vol_ratio_window = cfg["volume_ratio_window"]

    # Filtrar solo tickers equity para métricas de amplitud y correlación
    equity_tickers = _load_equity_tickers(config)
    equity_in_prices = [t for t in equity_tickers if t in prices]

    if not equity_in_prices:
        logger.warning(
            "No hay tickers equity en prices. Usando todos los tickers disponibles."
        )
        equity_in_prices = list(prices.keys())

    logger.debug(
        "Régimen: %d tickers equity de %d totales",
        len(equity_in_prices), len(prices),
    )

    # Retornos de acciones equity
    equity_prices = {t: prices[t] for t in equity_in_prices}
    returns_equity = build_returns_matrix(equity_prices)

    features = pd.DataFrame(index=returns_equity.index)

    # 1. Volatilidad realizada rolling (media de std individual sobre el universo equity)
    for w in vol_windows:
        col = f"vol_{w}d"
        # std rolling por ticker, luego promedio del universo
        rolling_std = returns_equity.rolling(window=w, min_periods=max(w // 2, 3)).std()
        features[col] = rolling_std.mean(axis=1)

    # 2. VIX: alinear al índice de retornos (forward-fill para festivos)
    vix_aligned = vix.reindex(returns_equity.index, method="ffill")
    features["vix"] = vix_aligned.values

    # 3. Correlación media rolling (solo equity)
    features["avg_correlation"] = compute_rolling_correlation(
        returns_equity.dropna(axis=1, how="all"),
        window=corr_window,
    )

    # 4. Volumen relativo: ratio volumen / MA20, promediado sobre equity
    vol_ratios = {}
    for ticker in equity_in_prices:
        vol = prices[ticker]["Volume"]
        ma_vol = vol.rolling(window=vol_ratio_window).mean()
        vol_ratios[ticker] = vol / ma_vol.replace(0, np.nan)
    if vol_ratios:
        vol_ratio_df = pd.DataFrame(vol_ratios)
        # Alinear al índice de features (retornos puede tener una fila menos)
        features["avg_volume_ratio"] = vol_ratio_df.reindex(features.index).mean(axis=1)

    # 5. Amplitud del mercado: % acciones equity al alza
    features["market_breadth"] = compute_market_breadth(returns_equity)

    logger.debug(
        "Features de régimen calculadas: %d fechas, %d columnas",
        len(features), len(features.columns),
    )
    return features


def compute_kelly_fraction_by_date(
    val_index: pd.DatetimeIndex,
    train_regime: pd.DataFrame,
    val_regime: pd.DataFrame,
    base_fraction: float,
    boost_fraction: float,
    vix_percentile: float,
    risk_scale: pd.Series,
    veto: pd.Series,
    require_normal_regime: bool = True,
) -> tuple[pd.Series, float]:
    """
    Kelly dinámico por fecha (Exp5): sube rho en días de régimen normal y VIX bajo.

    Condiciones para usar ``boost_fraction`` (anti-leakage: umbral VIX solo en train):
        - VIX del día < percentil ``vix_percentile`` del VIX en train
        - Si ``require_normal_regime``: risk_scale == 1.0 y veto binario == 0

    Args:
        val_index: Fechas de validación del backtest.
        train_regime: Features de régimen del período de entrenamiento.
        val_regime: Features de régimen del período de validación.
        base_fraction: rho base (ej. 0.5 Half-Kelly).
        boost_fraction: rho elevado en régimen normal (ej. 0.6).
        vix_percentile: Percentil del VIX en train para umbral (ej. 70).
        risk_scale: Serie Conspiranoico soft (1.0 = normal).
        veto: Serie veto binario del Conspiranoico.
        require_normal_regime: Si True, no boost en días con veto o Kelly reducido.

    Returns:
        (kelly_by_date, vix_threshold): Serie de rho por fecha y umbral VIX usado.
    """
    result = pd.Series(base_fraction, index=val_index, dtype=float)

    train_vix = train_regime["vix"].dropna()
    if train_vix.empty:
        logger.warning("Kelly boost: sin VIX en train. Usando rho base para todos los días.")
        return result, float("nan")

    vix_threshold = float(np.percentile(train_vix, vix_percentile))
    val_vix = val_regime["vix"].reindex(val_index)

    low_vix = val_vix < vix_threshold

    if require_normal_regime:
        scale_ok = risk_scale.reindex(val_index, fill_value=1.0) >= 1.0
        veto_ok = veto.reindex(val_index, fill_value=0) == 0
        boost_mask = low_vix.fillna(False) & scale_ok & veto_ok
    else:
        boost_mask = low_vix.fillna(False)

    result.loc[boost_mask] = boost_fraction

    n_boost = int(boost_mask.sum())
    logger.info(
        "Kelly boost — %d/%d días con rho=%.2f (VIX < %.2f, p%.0f train)",
        n_boost, len(val_index), boost_fraction, vix_threshold, vix_percentile,
    )

    return result, vix_threshold
