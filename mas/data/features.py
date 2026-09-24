"""
Cálculo de features técnicas sobre datos OHLCV diarios.

Todas las funciones son lookback-safe: el valor del día T solo usa datos
hasta T-1 (o T, para el propio precio de cierre). Nunca usan datos futuros.

Uso:
    from mas.data.features import compute_all_features
    features_df = compute_all_features(ohlcv_df)
"""

import pandas as pd
import numpy as np


# ── Indicadores individuales ──────────────────────────────────────────────────

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI — Relative Strength Index [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("rsi")


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD — Moving Average Convergence Divergence."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    })


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands: posición del precio dentro de las bandas."""
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "bb_pct_b": pct_b,
    })


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """ATR — Average True Range. Mide la volatilidad reciente."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=window - 1, min_periods=window).mean().rename("atr")


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volumen relativo vs media de N días. >1 = volumen elevado."""
    avg_volume = volume.rolling(window=window).mean()
    return (volume / avg_volume.replace(0, np.nan)).rename("volume_ratio")


def compute_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Retorno porcentual entre N días."""
    return close.pct_change(periods=periods).rename(f"return_{periods}d")


def compute_sma_ratio(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Ratio SMA rápida / SMA lenta. >1 = tendencia alcista."""
    sma_fast = close.rolling(window=fast).mean()
    sma_slow = close.rolling(window=slow).mean()
    return (sma_fast / sma_slow.replace(0, np.nan)).rename(f"sma_{fast}_{slow}_ratio")


# ── Pipeline completo ─────────────────────────────────────────────────────────

def feature_windows_from_config(config: dict) -> dict:
    """Extrae overrides de ventanas de indicadores (perfil long_term, etc.)."""
    return dict(config.get("matematico", {}).get("features_override") or {})


def compute_all_features(
    df: pd.DataFrame,
    windows: dict | None = None,
) -> pd.DataFrame:
    """
    Calcula todas las features técnicas sobre un DataFrame OHLCV.

    Args:
        df: DataFrame con columnas Open, High, Low, Close, Volume.
            El índice debe ser DatetimeIndex.

    Returns:
        DataFrame con todas las features. Las primeras filas tendrán NaN
        por los lookback periods de cada indicador — esto es correcto y esperado.

    IMPORTANTE ANTI-LEAKAGE:
        El target (retorno del día siguiente) se calcula aquí pero se asigna
        al día T para predecir T+1. El backtester es responsable de garantizar
        que el modelo solo ve features de T para predecir el retorno de T+1.
    """
    w = windows or {}
    close_norm_window = int(w.get("close_norm_window", 20))
    rsi_14_w = int(w.get("rsi_window", 14))
    rsi_28_w = int(w.get("rsi_window_long", 28))
    macd_windows = w.get("macd_windows", [12, 26, 9])
    bb_window = int(w.get("bollinger_window", 20))
    atr_w = int(w.get("atr_window", 14))
    vol_window = int(w.get("volume_window", 20))
    sma_pair = w.get("sma_windows", [20, 50])
    sma_fast = int(sma_pair[0])
    sma_slow = int(sma_pair[1]) if len(sma_pair) > 1 else 50
    sma_long_pair = w.get("sma_long_windows", [50, 200])
    sma_long_fast = int(sma_long_pair[0])
    sma_long_slow = int(sma_long_pair[1]) if len(sma_long_pair) > 1 else 200

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    features = pd.DataFrame(index=df.index)

    # Precios normalizados
    features["close"] = close
    features["close_norm"] = close / close.rolling(close_norm_window).mean()

    # Retornos históricos (lookback, no futuro)
    features["return_1d"] = compute_returns(close, 1)
    features["return_5d"] = compute_returns(close, 5)
    features["return_20d"] = compute_returns(close, 20)

    # RSI (nombres de columna fijos para el Matemático)
    features["rsi_14"] = compute_rsi(close, rsi_14_w)
    features["rsi_28"] = compute_rsi(close, rsi_28_w)

    # MACD
    macd_fast, macd_slow, macd_signal = (
        int(macd_windows[0]),
        int(macd_windows[1]),
        int(macd_windows[2]) if len(macd_windows) > 2 else 9,
    )
    macd_df = compute_macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    features = pd.concat([features, macd_df], axis=1)

    # Bollinger Bands
    bb_df = compute_bollinger_bands(close, window=bb_window)
    features = pd.concat([features, bb_df], axis=1)

    # ATR (volatilidad)
    features["atr"] = compute_atr(high, low, close, window=atr_w)
    features["atr_pct"] = features["atr"] / close

    # Volumen
    features["volume_ratio"] = compute_volume_ratio(volume, window=vol_window)

    # SMA ratios (tendencia) — nombres fijos, ventanas configurables
    features["sma_20_50_ratio"] = compute_sma_ratio(close, sma_fast, sma_slow)
    features["sma_50_200_ratio"] = compute_sma_ratio(close, sma_long_fast, sma_long_slow)

    # Target: retorno del día SIGUIENTE (shift(-1) = mirar un día hacia adelante)
    # ATENCIÓN: este campo solo se usa para entrenar, nunca como feature de entrada.
    # El último día siempre tiene NaN (no hay T+1) — se excluye en Matematico._clean().
    features["target_return_1d"] = close.pct_change(1).shift(-1)
    features["target_binary"] = np.where(
        features["target_return_1d"].isna(),
        np.nan,
        (features["target_return_1d"] > 0).astype(float),
    )

    return features
