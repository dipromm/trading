"""
Pydantic v2 schemas for the MAS Trading System API.

All response models are defined here as the single source of truth.
The frontend generates TypeScript types from the OpenAPI spec produced by these models.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Composable sub-models ────────────────────────────────────────────────────


class KellyInputs(BaseModel):
    p: float = Field(description="Win probability from calibrated model")
    b: float = Field(description="Win/loss ratio")
    rho: float = Field(description="Kelly fraction multiplier (0.5 for half-Kelly)")


class AgentVotes(BaseModel):
    matematico: float = Field(description="Calibrated probability from El Matematico")
    analista: Optional[float] = Field(default=None, description="Sentiment probability from El Analista")
    cazador: Literal["signal", "silence"] = Field(description="Insider alert status from El Cazador")
    conspiranoico: Literal["active", "inactive"] = Field(description="Regime veto status")


# ── Paper Trading ────────────────────────────────────────────────────────────


class PaperTradingLog(BaseModel):
    date: date
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    probability: float
    kelly_fraction: float
    kelly_inputs: KellyInputs
    agent_votes: AgentVotes
    reason: str
    position_size_eur: float


# ── Equity Curve ─────────────────────────────────────────────────────────────


class EquityCurvePoint(BaseModel):
    date: date
    portfolio_value: float
    buyhold_value: Optional[float] = None
    sma_value: Optional[float] = None
    period: Literal["walkforward", "holdout", "paper_trading"]


# ── Metrics ──────────────────────────────────────────────────────────────────


class MetricsResponse(BaseModel):
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    n_trading_days: int
    sortino_ratio: Optional[float] = None
    profit_factor: Optional[float] = None


# ── Walk-Forward Windows ─────────────────────────────────────────────────────


class WalkForwardWindow(BaseModel):
    iteration: int
    val_start: str
    val_end: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    n_tickers: int


# ── Registry / Status ────────────────────────────────────────────────────────


class ModelFiles(BaseModel):
    matematico: str
    juez: str


class RegistryStatus(BaseModel):
    experiment_id: str
    trained_at: Optional[str] = None
    model_hash_sha256: Optional[dict[str, str]] = None
    val_sharpe: float
    val_max_drawdown: float
    holdout_start: str
    last_run_at: Optional[str] = None
    last_run_status: Optional[str] = None
    files: ModelFiles


# ── Positions ────────────────────────────────────────────────────────────────


class PositionEntry(BaseModel):
    ticker: str
    quantity: float
    entry_price: float
    entry_date: date
    current_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class PositionsResponse(BaseModel):
    positions: list[PositionEntry]
    total_value: float
    cash: float
    as_of: Optional[date] = None


# ── Experiments ──────────────────────────────────────────────────────────────


class ExperimentSummary(BaseModel):
    experiment_id: str
    timestamp: Optional[str] = None
    total_return_pct: Optional[float] = None
    annualized_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    calmar_ratio: Optional[float] = None
    win_rate_pct: Optional[float] = None
    n_trading_days: Optional[int] = None
    n_windows: Optional[int] = None


# ── Council Verdict ──────────────────────────────────────────────────────────


class CouncilVerdict(BaseModel):
    date: date
    decisions: list[PaperTradingLog]


# ── Per-period metrics (WF vs Holdout) ───────────────────────────────────────


class BaselineMetrics(BaseModel):
    total_return_pct: Optional[float] = None
    annualized_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    calmar_ratio: Optional[float] = None
    win_rate_pct: Optional[float] = None
    n_trading_days: Optional[int] = None


class PeriodMetricsResponse(BaseModel):
    mas: MetricsResponse
    buy_and_hold: BaselineMetrics
    sma_crossover: BaselineMetrics


# ── Market Close ─────────────────────────────────────────────────────────────


class MarketCloseResponse(BaseModel):
    last_market_close: str = Field(description="Human-readable last market close date")
    pipeline_ran_at: Optional[str] = Field(default=None, description="Last pipeline execution time")


# ── Generic status for not-started endpoints ─────────────────────────────────


class StatusMessage(BaseModel):
    status: str
