"""
Baseline 2: SMA Crossover (20/50).

Compra cuando la media móvil de 20 días cruza por encima de la de 50 días.
Vende cuando cruza por debajo.

Estrategia de equal weight independiente por ticker: cada ticker gestiona
su propio sub-portfolio con 1/N del capital inicial. Esto evita rebalanceos
constantes entre tickers manteniendo la comparabilidad con Buy & Hold.

Este es el mínimo bar técnico a superar para un sistema ML.
Si el Matemático solo no supera este baseline en walk-forward,
los features técnicos no tienen poder predictivo suficiente.

Nota sobre lookback: los primeros `slow_window` días no generan señal
porque las medias móviles no tienen datos suficientes. Durante ese
período el capital permanece en efectivo (sin comisiones).
"""

import logging
from typing import Optional

import pandas as pd

from mas.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def _simulate_ticker(
    close_series: pd.Series,
    capital: float,
    commission_pct: float,
    fast_window: int,
    slow_window: int,
) -> tuple[pd.Series, int]:
    """
    Simula SMA Crossover para un único ticker con capital independiente.

    El capital inicia en efectivo. Se invierte todo al primer cruce alcista
    y se liquida al primer cruce bajista (con comisión en cada operación).

    Args:
        close_series: Precios de cierre sin NaN, indexados por fecha.
        capital: Capital inicial asignado a este ticker.
        commission_pct: Comisión por operación (ej. 0.0008 = 0.08%).
        fast_window: Ventana de la media rápida (días).
        slow_window: Ventana de la media lenta (días).

    Returns:
        (sub_equity_curve, n_trades): Serie con el valor del sub-portfolio
        día a día y número total de operaciones (compras + ventas).
    """
    sma_fast = close_series.rolling(fast_window).mean()
    sma_slow = close_series.rolling(slow_window).mean()

    # True cuando la media rápida está por encima de la lenta.
    # NaN en los primeros slow_window días → False (sin posición).
    signal = (sma_fast > sma_slow).fillna(False)

    cash = capital
    shares = 0.0
    in_position = False
    n_trades = 0
    values: list[float] = []

    for i in range(len(close_series)):
        price = close_series.iloc[i]
        sig = bool(signal.iloc[i])

        if sig and not in_position:
            # Cruce alcista: COMPRA con todo el efectivo disponible
            commission = cash * commission_pct
            shares = (cash - commission) / price
            cash = 0.0
            in_position = True
            n_trades += 1

        elif not sig and in_position:
            # Cruce bajista: VENTA de todas las acciones
            sale_value = shares * price
            commission = sale_value * commission_pct
            cash = sale_value - commission
            shares = 0.0
            in_position = False
            n_trades += 1

        values.append(cash + shares * price)

    return pd.Series(values, index=close_series.index), n_trades


def run(
    prices: dict[str, pd.DataFrame],
    fast_window: int = 20,
    slow_window: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[dict] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Simula SMA Crossover sobre el universo completo con equal weight por ticker.

    Cada ticker recibe 1/N del capital inicial y opera de forma independiente.
    La curva de capital total es la suma de todos los sub-portfolios.

    Args:
        prices: dict {ticker: DataFrame OHLCV con columna "Close"}
        fast_window: Ventana de la media rápida en días (por defecto 20).
        slow_window: Ventana de la media lenta en días (por defecto 50).
        start_date: Fecha de inicio del período de evaluación (YYYY-MM-DD).
                    IMPORTANTE: los datos deben incluir al menos `slow_window`
                    días anteriores a esta fecha para que las SMAs sean válidas
                    desde el primer día de evaluación.
        end_date: Fecha de fin del período de evaluación (YYYY-MM-DD, exclusivo).
        config: Configuración del proyecto. Si es None, se carga desde config.yaml.

    Returns:
        (equity_curve, daily_returns): Series indexadas por fecha.
    """
    if config is None:
        config = load_config()

    commission_pct: float = config["backtester"]["commission_pct"]
    initial_capital: float = config["backtester"]["initial_capital"]
    tickers = list(prices.keys())

    if not tickers:
        logger.error("SMA Crossover: el diccionario de precios está vacío.")
        empty = pd.Series(dtype=float)
        return empty, empty

    # Construir DataFrame de cierres alineados con forward-fill de huecos
    close = pd.DataFrame({t: prices[t]["Close"] for t in tickers})
    close = close.ffill()

    # Filtrar ventana de evaluación DESPUÉS de calcular las SMAs para no
    # perder el lookback necesario. Ver nota en el docstring del módulo.
    eval_close = close.copy()
    if start_date:
        eval_close = eval_close[eval_close.index >= start_date]
    if end_date:
        eval_close = eval_close[eval_close.index < end_date]
    eval_close = eval_close.dropna(how="all")

    if eval_close.empty:
        logger.error("SMA Crossover: DataFrame vacío después de filtrar fechas.")
        empty = pd.Series(dtype=float)
        return empty, empty

    n_tickers = len(tickers)
    capital_per_ticker = initial_capital / n_tickers
    total_trades = 0

    # Simular cada ticker de forma independiente usando la serie COMPLETA
    # (para que las SMAs arranquen con datos históricos suficientes)
    # y luego recortar al período de evaluación.
    sub_portfolios: list[pd.Series] = []
    tickers_skipped = 0

    for ticker in tickers:
        full_series = close[ticker].dropna()

        if len(full_series) < slow_window:
            logger.warning(
                "SMA Crossover: %s tiene solo %d días de datos (mínimo %d). Ticker omitido.",
                ticker, len(full_series), slow_window,
            )
            # Su capital queda en efectivo durante todo el período
            sub_portfolios.append(
                pd.Series(capital_per_ticker, index=eval_close.index)
            )
            tickers_skipped += 1
            continue

        sub_equity, n_trades = _simulate_ticker(
            close_series=full_series,
            capital=capital_per_ticker,
            commission_pct=commission_pct,
            fast_window=fast_window,
            slow_window=slow_window,
        )
        total_trades += n_trades

        # Recortar al período de evaluación y reindexar (ffill por festivos)
        sub_equity = sub_equity.reindex(eval_close.index, method="ffill")
        sub_portfolios.append(sub_equity)

    # Sumar todos los sub-portfolios para obtener la curva total
    equity_curve = pd.concat(sub_portfolios, axis=1).sum(axis=1)
    equity_curve.name = f"sma_crossover_{fast_window}_{slow_window}"

    daily_returns = equity_curve.pct_change().fillna(0)

    logger.info(
        "SMA Crossover (%d/%d): %d días | %d tickers (%d omitidos) | "
        "%d operaciones | %.0f€ → %.0f€",
        fast_window, slow_window,
        len(equity_curve),
        n_tickers - tickers_skipped, tickers_skipped,
        total_trades,
        initial_capital, equity_curve.iloc[-1],
    )

    return equity_curve, daily_returns
