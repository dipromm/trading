"""
Baseline 2: SMA Crossover (20/50).

Compra cuando la media móvil de 20 días cruza por encima de la de 50 días.
Vende cuando cruza por debajo.

Este es el mínimo bar técnico a superar para un sistema ML.
Si el Matemático solo no supera este baseline en walk-forward,
los features técnicos no tienen poder predictivo suficiente.
"""

import logging
from typing import Optional

import pandas as pd

from utils.config_loader import load_config

logger = logging.getLogger(__name__)


def run(
    prices: dict[str, pd.DataFrame],
    fast_window: int = 20,
    slow_window: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[dict] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Simula SMA Crossover sobre el universo completo con equal weight.

    La señal es independiente por ticker: cada uno tiene su propio cruce de medias.

    Args:
        prices: dict {ticker: DataFrame OHLCV}
        fast_window: Ventana de la media rápida (días)
        slow_window: Ventana de la media lenta (días)
        start_date: Fecha inicio del período
        end_date: Fecha fin del período
        config: Configuración del proyecto

    Returns:
        (equity_curve, daily_returns): Series indexadas por fecha.
    """
    # TODO: implementar en Fase 2
    # 1. Calcular SMA fast y slow para cada ticker
    # 2. Señal = 1 si SMA_fast > SMA_slow, else 0
    # 3. Detectar cruces: comprar en cruce alcista, vender en cruce bajista
    # 4. Aplicar comisiones en cada cambio de señal
    # 5. Calcular curva de capital con equal weight entre tickers con señal activa
    raise NotImplementedError("Implementar en Fase 2")
