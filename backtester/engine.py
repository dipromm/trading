"""
Motor de Backtesting.

Simula operaciones históricas día a día, aplicando comisiones reales,
y registra un log auditable de cada operación.

CRÍTICO: Este es el componente más importante del proyecto.
Un bug aquí invalida todos los resultados. Implementar y testear primero
(Fase 2), antes de entrenar ningún agente.

Ver tests/test_backtester.py para los checks obligatorios antes de usar.

Flujo por día T:
    1. Recibir señales de los agentes para el día T
    2. Si El Conspiranoico veta: no abrir posiciones nuevas
    3. Para cada ticker con señal del Juez:
       a. El Gestor de Riesgos calcula f* (Fractional Kelly)
       b. Si f* > 0: generar orden de COMPRA (si no hay posición)
       c. Si f* <= 0: generar orden de VENTA (si hay posición abierta)
       d. Verificar stop-loss en posiciones existentes
    4. Ejecutar órdenes al precio de cierre del día T
       (o apertura de T+1 si se simula un delay realista)
    5. Registrar operación en log JSON auditable
    6. Actualizar curva de capital
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from backtester.metrics import summary
from utils.config_loader import load_config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Representa una posición abierta en un ticker."""
    ticker: str
    entry_price: float
    entry_date: date
    shares: float
    capital_invested: float


@dataclass
class TradeLog:
    """Log auditable de una operación individual."""
    date: str
    ticker: str
    action: str                     # "BUY" o "SELL"
    price: float
    shares: float
    capital: float
    commission: float
    kelly_fraction: float
    agent_votes: dict = field(default_factory=dict)
    reason: str = ""                # "signal", "stop_loss", "end_of_period"


class BacktestEngine:
    """
    Motor de backtesting con log auditable y simulación de comisiones.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()

        self.initial_capital = config["backtester"]["initial_capital"]
        self.commission_pct = config["backtester"]["commission_pct"]
        self.log_dir = Path(config["general"]["log_dir"]) / "trades"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.capital: float = self.initial_capital
        self.positions: dict[str, Position] = {}
        self.equity_curve: list[float] = []
        self.trade_logs: list[TradeLog] = []

    def reset(self) -> None:
        """Reinicia el estado para una nueva ventana walk-forward."""
        self.capital = self.initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trade_logs = []

    def portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Valor total del portfolio (capital en efectivo + posiciones abiertas)."""
        positions_value = sum(
            pos.shares * current_prices.get(pos.ticker, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.capital + positions_value

    def run(
        self,
        prices: dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        kelly_fractions: pd.DataFrame,
        veto_signal: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Ejecuta el backtest sobre un período completo.

        Args:
            prices: dict {ticker: OHLCV DataFrame}
            signals: DataFrame {ticker: señal del Juez ∈ [0,1]} por fecha
            kelly_fractions: DataFrame {ticker: f* del Gestor} por fecha
            veto_signal: Serie binaria del Conspiranoico por fecha (1=veto)

        Returns:
            (equity_curve, daily_returns): Series indexadas por fecha.
        """
        # TODO: implementar en Fase 2
        # Para cada fecha en signals.index:
        #   1. Si veto_signal[fecha] == 1: skip (no abrir posiciones)
        #   2. Para cada ticker:
        #      a. Verificar stop-loss en posiciones abiertas
        #      b. Si kelly_fractions[ticker][fecha] <= 0 y hay posición: vender
        #      c. Si kelly_fractions[ticker][fecha] > 0 y no hay posición: comprar
        #   3. Aplicar comisión en cada operación
        #   4. Registrar TradeLog
        #   5. Calcular portfolio_value y añadir a equity_curve
        raise NotImplementedError("Implementar en Fase 2 — ANTES de entrenar cualquier agente")

    def save_trade_logs(self, filename: str = "trades.jsonl") -> Path:
        """Guarda el log de operaciones en formato JSONL (una operación por línea)."""
        output_path = self.log_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            for log in self.trade_logs:
                f.write(json.dumps(log.__dict__) + "\n")
        logger.info(f"Log de operaciones guardado: {output_path} ({len(self.trade_logs)} operaciones)")
        return output_path
