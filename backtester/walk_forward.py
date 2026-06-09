"""
Validación Walk-Forward.

Implementa la estrategia de validación temporal que garantiza resultados
estadísticamente creíbles y sin data leakage entre entrenamientos.

Estructura de ventanas (con config por defecto: 3 años train, 1 año val):

    ITER 1: ENTRENA [2018-2020] ──────────────── VALIDA [2021]
    ITER 2: ENTRENA [2018-2021] ─────────────────────── VALIDA [2022]
    ITER 3: ENTRENA [2018-2022] ────────────────────────────── VALIDA [2023]
    ITER 4: ENTRENA [2018-2023] ─────────────────────────────────── VALIDA [2024]
    HOLDOUT [2025]: ████████████████ NO TOCAR hasta que el sistema esté finalizado

Las predicciones de validación de todas las iteraciones se concatenan para
obtener ~4 años de predicciones out-of-sample sobre las que calcular las métricas finales.

NOTA sobre el Juez:
    En la Iteración 1, el Juez no tiene datos previos de agentes para entrenarse.
    El Juez en la Iteración 1 actúa como pass-through del Matemático.
    En Iteraciones 2+, el Juez se entrena con las predicciones de los agentes
    de la iteración anterior.
"""

import logging
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from utils.config_loader import load_config

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Una ventana individual de entrenamiento + validación."""
    iteration: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str


def generate_windows(config: dict | None = None) -> list[WalkForwardWindow]:
    """
    Genera las ventanas walk-forward según la configuración.

    Args:
        config: Si None, carga config.yaml

    Returns:
        Lista de WalkForwardWindow ordenadas cronológicamente.
    """
    if config is None:
        config = load_config()

    start = pd.Timestamp(config["data"]["start_date"])
    holdout_start = pd.Timestamp(config["data"]["holdout_start"])
    train_years = config["backtester"]["walk_forward_train_years"]
    val_years = config["backtester"]["walk_forward_val_years"]

    windows = []
    iteration = 1
    val_start = start + pd.DateOffset(years=train_years)

    while val_start < holdout_start:
        val_end = val_start + pd.DateOffset(years=val_years)
        if val_end > holdout_start:
            val_end = holdout_start

        windows.append(WalkForwardWindow(
            iteration=iteration,
            train_start=start.strftime("%Y-%m-%d"),
            train_end=val_start.strftime("%Y-%m-%d"),
            val_start=val_start.strftime("%Y-%m-%d"),
            val_end=val_end.strftime("%Y-%m-%d"),
        ))

        iteration += 1
        val_start = val_end

    logger.info(f"Walk-forward: {len(windows)} ventanas generadas")
    for w in windows:
        logger.info(f"  Iter {w.iteration}: train [{w.train_start} → {w.train_end}] | val [{w.val_start} → {w.val_end}]")

    return windows


class WalkForwardValidator:
    """
    Orquestador del proceso walk-forward completo.

    Uso:
        validator = WalkForwardValidator()
        results = validator.run(agents, judge, gestor, prices, features)
        # results contiene equity_curve y métricas de todo el período out-of-sample
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config
        self.windows = generate_windows(config)

    def run(self, agents, judge, gestor, prices, features) -> dict:
        """
        Ejecuta el proceso walk-forward completo sobre todos los agentes.

        Returns:
            dict con equity_curve concatenada de todos los períodos de validación,
            métricas agregadas, y métricas por ventana.
        """
        # TODO: implementar en Fase 2/3
        # Para cada window en self.windows:
        #   1. Filtrar datos de entrenamiento y validación
        #   2. Llamar a agent.fit() para cada agente con datos de train
        #   3. Llamar a agent.predict() sobre datos de val
        #   4. Pasar señales al BacktestEngine
        #   5. Guardar resultados de la ventana
        # Concatenar equity_curves de todas las ventanas
        # Calcular métricas finales con backtester.metrics.summary()
        raise NotImplementedError("Implementar en Fase 2")
