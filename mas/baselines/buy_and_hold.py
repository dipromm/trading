"""
Baseline 1: Buy & Hold Equal Weight.

Compra todas las acciones del universo en partes iguales el primer día
y no toca nada hasta el final del período.

Este es el benchmark principal: cualquier estrategia activa debe superar
su Sharpe Ratio para justificar la complejidad añadida.

Si el sistema MAS no supera Buy & Hold en walk-forward, no añade valor.
"""

import logging
from typing import Optional

import pandas as pd

from mas.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def run(
    prices: dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[dict] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Simula Buy & Hold sobre el universo completo.

    Args:
        prices: dict {ticker: DataFrame con columna 'Close'}
        start_date: Fecha inicio del período a evaluar (YYYY-MM-DD)
        end_date: Fecha fin del período a evaluar (YYYY-MM-DD)
        config: Configuración del proyecto

    Returns:
        (equity_curve, daily_returns): Series indexadas por fecha.
    """
    if config is None:
        config = load_config()

    commission = config["backtester"]["commission_pct"]
    initial_capital = config["backtester"]["initial_capital"]
    tickers = list(prices.keys())

    # Construir DataFrame de cierres alineados
    close = pd.DataFrame({t: prices[t]["Close"] for t in tickers})
    if start_date:
        close = close[close.index >= start_date]
    if end_date:
        close = close[close.index < end_date]
    close = close.dropna(how="all").ffill()

    if close.empty:
        logger.error("Buy & Hold: DataFrame de precios vacío después de filtrar fechas.")
        empty = pd.Series(dtype=float)
        return empty, empty

    n_tickers = len(tickers)
    capital_per_ticker = initial_capital / n_tickers

    # Comprar el primer día con comisión de entrada
    first_day = close.index[0]
    shares: dict[str, float] = {}
    cash = 0.0

    for ticker in tickers:
        price = close.loc[first_day, ticker]
        if pd.isna(price) or price <= 0:
            cash += capital_per_ticker
            logger.warning(f"[Buy&Hold] Precio inválido para {ticker} el {first_day}. Capital reasignado a efectivo.")
            continue
        capital_after_commission = capital_per_ticker * (1 - commission)
        shares[ticker] = capital_after_commission / price

    # Calcular curva de capital día a día
    equity_values = []
    for date in close.index:
        portfolio_value = cash
        for ticker, n_shares in shares.items():
            price = close.loc[date, ticker]
            if not pd.isna(price) and price > 0:
                portfolio_value += n_shares * price
        equity_values.append(portfolio_value)

    equity_curve = pd.Series(equity_values, index=close.index, name="buy_and_hold")
    daily_returns = equity_curve.pct_change().dropna()

    logger.info(
        f"Buy & Hold: {len(equity_curve)} días | "
        f"Capital inicial: {initial_capital:.0f}€ → Final: {equity_curve.iloc[-1]:.0f}€"
    )
    return equity_curve, daily_returns
