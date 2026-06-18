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

import pandas as pd

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

        self.initial_capital: float = config["backtester"]["initial_capital"]
        self.commission_pct: float = config["backtester"]["commission_pct"]
        self.atr_multiplier: float = config["risk_manager"]["stop_loss_atr_multiplier"]
        self.min_kelly_threshold: float = config["risk_manager"]["min_kelly_threshold"]
        self.max_position_pct: float = config["risk_manager"]["max_position_pct"]
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
        atr_data: pd.DataFrame | None = None,
        agent_votes_log: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Ejecuta el backtest sobre un período completo.

        Args:
            prices: dict {ticker: OHLCV DataFrame con columna "close"}
            signals: DataFrame {ticker: señal del Juez ∈ [0,1]} indexado por fecha
            kelly_fractions: DataFrame {ticker: f* del Gestor} indexado por fecha
            veto_signal: Serie binaria del Conspiranoico por fecha (1=veto activo)
            atr_data: DataFrame {ticker: ATR} indexado por fecha.
                      Si es None, el stop-loss dinámico queda desactivado.
            agent_votes_log: DataFrame {ticker: dict de votos de cada agente} por fecha.
                             Si se pasa, se incluye en cada TradeLog para trazabilidad XAI.

        Returns:
            (equity_curve, daily_returns): Series indexadas por fecha.
        """
        dates = signals.index

        for fecha in dates:
            current_prices: dict[str, float] = {
                ticker: float(prices[ticker].loc[fecha, "Close"])
                for ticker in prices
                if fecha in prices[ticker].index
            }

            if not current_prices:
                self.equity_curve.append(self.portfolio_value({}))
                continue

            hay_veto = veto_signal.get(fecha, 0) == 1

            # Paso 1: verificar stop-loss en posiciones abiertas (antes de nuevas órdenes)
            if atr_data is not None and fecha in atr_data.index:
                for ticker in list(self.positions.keys()):
                    if ticker not in current_prices or ticker not in atr_data.columns:
                        continue
                    atr = atr_data.loc[fecha, ticker]
                    if not pd.isna(atr):
                        self._check_stop_loss(fecha, ticker, current_prices[ticker], float(atr))

            # Paso 2: procesar señales del Juez para cada ticker
            for ticker in kelly_fractions.columns:
                if ticker not in current_prices:
                    continue

                if fecha not in kelly_fractions.index:
                    continue

                price = current_prices[ticker]
                f_star = kelly_fractions.loc[fecha, ticker]

                if pd.isna(f_star):
                    f_star = 0.0

                votes: dict = {}
                if (
                    agent_votes_log is not None
                    and fecha in agent_votes_log.index
                    and ticker in agent_votes_log.columns
                ):
                    votes = agent_votes_log.loc[fecha, ticker] or {}

                if f_star <= 0 and ticker in self.positions:
                    self._sell(fecha, ticker, price, f_star=0.0, reason="signal", agent_votes=votes)

                elif f_star > self.min_kelly_threshold and ticker not in self.positions and not hay_veto:
                    self._buy(fecha, ticker, price, f_star, current_prices, agent_votes=votes)

            valor = self.portfolio_value(current_prices)
            self.equity_curve.append(valor)

        equity = pd.Series(self.equity_curve, index=dates)
        daily_returns = equity.pct_change().fillna(0)
        return equity, daily_returns

    def close_all_positions(self, fecha, current_prices: dict[str, float]) -> None:
        """
        Cierra todas las posiciones abiertas al final de un período walk-forward.

        Llamar al final de cada ventana de validación antes de hacer reset(),
        para que las posiciones no queden abiertas entre ventanas.
        """
        for ticker in list(self.positions.keys()):
            price = current_prices.get(ticker, self.positions[ticker].entry_price)
            self._sell(fecha, ticker, price, f_star=0.0, reason="end_of_period")

    def _buy(
        self,
        fecha,
        ticker: str,
        price: float,
        f_star: float,
        current_prices: dict[str, float],
        agent_votes: dict | None = None,
    ) -> None:
        # Red de seguridad: no superar el cap máximo definido en config
        f_star = min(f_star, self.max_position_pct)

        portfolio_val = self.portfolio_value(current_prices)
        capital_to_invest = portfolio_val * f_star

        # No invertir más de lo que hay en efectivo
        capital_to_invest = min(capital_to_invest, self.capital)

        if capital_to_invest <= 0:
            return

        commission_cost = capital_to_invest * self.commission_pct
        capital_after_commission = capital_to_invest - commission_cost
        shares = capital_after_commission / price

        self.capital -= capital_to_invest
        self.positions[ticker] = Position(
            ticker=ticker,
            entry_price=price,
            entry_date=fecha,
            shares=shares,
            capital_invested=capital_after_commission,
        )
        self.trade_logs.append(TradeLog(
            date=str(fecha),
            ticker=ticker,
            action="BUY",
            price=price,
            shares=shares,
            capital=self.capital,
            commission=commission_cost,
            kelly_fraction=f_star,
            agent_votes=agent_votes or {},
            reason="signal",
        ))
        logger.debug(
            "BUY  %s @ %.2f  shares=%.4f  f*=%.3f  capital_restante=%.2f",
            ticker, price, shares, f_star, self.capital,
        )

    def _sell(
        self,
        fecha,
        ticker: str,
        price: float,
        f_star: float,
        reason: str,
        agent_votes: dict | None = None,
    ) -> None:
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        sale_value = pos.shares * price
        commission_cost = sale_value * self.commission_pct
        net_proceeds = sale_value - commission_cost

        self.capital += net_proceeds
        del self.positions[ticker]
        self.trade_logs.append(TradeLog(
            date=str(fecha),
            ticker=ticker,
            action="SELL",
            price=price,
            shares=pos.shares,
            capital=self.capital,
            commission=commission_cost,
            kelly_fraction=f_star,
            agent_votes=agent_votes or {},
            reason=reason,
        ))
        logger.debug(
            "SELL %s @ %.2f  reason=%s  capital=%.2f",
            ticker, price, reason, self.capital,
        )

    def _check_stop_loss(self, fecha, ticker: str, price: float, atr: float) -> None:
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        perdida = (price - pos.entry_price) / pos.entry_price  # negativo si ha bajado
        stop_loss_threshold = -(self.atr_multiplier * atr / pos.entry_price)

        if perdida < stop_loss_threshold:
            logger.info(
                "STOP-LOSS %s: pérdida=%.2f%%  umbral=%.2f%%",
                ticker, perdida * 100, stop_loss_threshold * 100,
            )
            self._sell(fecha, ticker, price, f_star=0.0, reason="stop_loss")

    def save_trade_logs(self, filename: str = "trades.jsonl") -> Path:
        """Guarda el log de operaciones en formato JSONL (una operación por línea)."""
        output_path = self.log_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            for log in self.trade_logs:
                f.write(json.dumps(log.__dict__) + "\n")
        logger.info(
            "Log de operaciones guardado: %s (%d operaciones)",
            output_path, len(self.trade_logs),
        )
        return output_path
