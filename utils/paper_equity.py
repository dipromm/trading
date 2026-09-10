"""
Paper trading equity tracking for the dashboard equity curve.

run_daily.py appends daily portfolio values to logs/dashboard/paper_equity.csv.
generate_dashboard_data.py merges that series into equity_curve.csv.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PAPER_EQUITY_PATH = ROOT / "logs" / "dashboard" / "paper_equity.csv"


def _close_on_date(
    prices: dict[str, pd.DataFrame],
    ticker: str,
    target_date: date,
) -> float | None:
    df = prices.get(ticker)
    if df is None or df.empty:
        return None
    ts = pd.Timestamp(target_date)
    idx = pd.to_datetime(df.index).normalize()
    subset = df.loc[idx <= ts]
    if subset.empty:
        return None
    return float(subset["Close"].iloc[-1])


def _position_market_value(
    ticker: str,
    pos: dict,
    prices: dict[str, pd.DataFrame],
    target_date: date,
) -> float:
    allocated = float(pos.get("allocated_eur") or 0)
    close = _close_on_date(prices, ticker, target_date)
    if close is None or allocated <= 0:
        return allocated

    entry_date_raw = pos.get("entry_date")
    if entry_date_raw:
        try:
            entry_d = date.fromisoformat(str(entry_date_raw)[:10])
            entry_close = _close_on_date(prices, ticker, entry_d)
            if entry_close and entry_close > 0:
                return allocated * (close / entry_close)
        except (TypeError, ValueError):
            pass

    qty = float(pos.get("quantity") or 0)
    entry_price = float(pos.get("entry_price") or 0)
    if qty > 0 and entry_price > 0:
        return qty * close

    return allocated


def compute_portfolio_value(
    positions_state: dict,
    prices: dict[str, pd.DataFrame],
    target_date: date,
    cfg: dict,
) -> float:
    """Mark-to-market portfolio value for a single date."""
    initial_capital = float(cfg["backtester"]["initial_capital"])
    raw_positions = positions_state.get("positions") or {}
    cash = max(0.0, float(positions_state.get("cash") or 0))

    # Normalise to {ticker: dict} — the list format is the API's presentation layer
    if isinstance(raw_positions, list):
        positions = {p["ticker"]: p for p in raw_positions if isinstance(p, dict) and p.get("ticker")}
    else:
        positions = dict(raw_positions)

    if not positions:
        # All-cash portfolio
        if cash > 0:
            return cash
        if PAPER_EQUITY_PATH.exists():
            pe = pd.read_csv(PAPER_EQUITY_PATH, parse_dates=[0], index_col=0)
            if not pe.empty:
                return float(pe["equity"].iloc[-1])
        return initial_capital

    invested = sum(
        _position_market_value(ticker, pos, prices, target_date)
        for ticker, pos in positions.items()
    )
    return cash + invested


def append_paper_equity_row(target_date: date, portfolio_value: float) -> None:
    """Upsert one row in logs/dashboard/paper_equity.csv."""
    PAPER_EQUITY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if PAPER_EQUITY_PATH.exists():
        df = pd.read_csv(PAPER_EQUITY_PATH, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
    else:
        df = pd.DataFrame(columns=["equity"])

    ts = pd.Timestamp(target_date).normalize()
    df.loc[ts, "equity"] = round(portfolio_value, 4)
    df = df.sort_index()
    df.to_csv(PAPER_EQUITY_PATH)
    logger.info(
        "Paper equity updated: %s -> %.2f EUR (%s)",
        target_date,
        portfolio_value,
        PAPER_EQUITY_PATH,
    )


def load_positions_state() -> dict:
    path = ROOT / "logs" / "positions_current.json"
    if not path.exists():
        return {"positions": {}, "cash": 0.0}
    return json.loads(path.read_text(encoding="utf-8"))


def get_cash_balance(current: dict, cfg: dict) -> float:
    """Resolve available cash from positions state or last paper equity row."""
    initial_capital = float(cfg["backtester"]["initial_capital"])
    cash = current.get("cash")
    if cash is not None and float(cash) >= 0:
        return float(cash)

    portfolio_value = current.get("portfolio_value_eur")
    positions = current.get("positions") or {}
    if isinstance(positions, dict) and portfolio_value is not None:
        allocated = sum(float(p.get("allocated_eur") or 0) for p in positions.values())
        return max(0.0, float(portfolio_value) - allocated)

    if PAPER_EQUITY_PATH.exists():
        pe = pd.read_csv(PAPER_EQUITY_PATH, parse_dates=[0], index_col=0)
        if not pe.empty:
            return float(pe["equity"].iloc[-1])

    return initial_capital


def apply_decisions_with_cash(
    positions: dict,
    cash: float,
    decisions: list[dict],
) -> tuple[dict, float]:
    """Apply BUY/SELL respecting available cash (BUY capped to cash on hand)."""
    for d in decisions:
        ticker = d.get("ticker")
        if not ticker:
            continue
        action = d.get("action")
        if action == "BUY" and ticker not in positions:
            requested = float(d.get("position_size_eur") or 0)
            if requested <= 0 or cash <= 0:
                continue
            allocated = min(requested, cash)
            positions[ticker] = {
                "quantity": 1,
                "entry_price": 0,
                "entry_date": d.get("date"),
                "allocated_eur": round(allocated, 2),
            }
            cash = round(cash - allocated, 2)
        elif action == "SELL" and ticker in positions:
            cash = round(cash + float(positions[ticker].get("allocated_eur") or 0), 2)
            del positions[ticker]
    return positions, cash


def apply_decisions_to_positions(
    positions: dict,
    decisions: list[dict],
    cash: float,
) -> tuple[dict, float]:
    """Backwards-compatible wrapper used by backfill."""
    return apply_decisions_with_cash(positions, cash, decisions)


def build_positions_state(
    positions: dict,
    cash: float,
    target_date: date,
) -> dict:
    total_allocated = sum(float(p.get("allocated_eur") or 0) for p in positions.values())
    cash = max(0.0, float(cash))
    return {
        "as_of": str(target_date),
        "positions": positions,
        "total_allocated": round(total_allocated, 2),
        "cash": round(cash, 2),
        "portfolio_value_eur": round(cash + total_allocated, 2),
        "n_positions": len(positions),
    }


def load_backfill_seed(cfg: dict, before_date: date) -> tuple[dict, float]:
    """Cash and positions at the end of the last day strictly before ``before_date``."""
    initial_capital = float(cfg["backtester"]["initial_capital"])
    positions: dict = {}
    cash = initial_capital

    if PAPER_EQUITY_PATH.exists():
        pe = pd.read_csv(PAPER_EQUITY_PATH, parse_dates=[0], index_col=0)
        pe.index = pd.to_datetime(pe.index).normalize()
        prior = pe[pe.index < pd.Timestamp(before_date)]
        if not prior.empty:
            cash = float(prior["equity"].iloc[-1])

    pos_path = ROOT / "logs" / "positions_current.json"
    if pos_path.exists():
        try:
            data = json.loads(pos_path.read_text(encoding="utf-8"))
            as_of = data.get("as_of")
            if as_of and date.fromisoformat(str(as_of)[:10]) < before_date:
                raw = data.get("positions") or {}
                if isinstance(raw, dict):
                    positions = dict(raw)
                cash = get_cash_balance(data, cfg)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    return positions, cash


def refresh_dashboard_equity_curve() -> None:
    """Regenerate logs/dashboard/equity_curve.csv (paper section included)."""
    from scripts.generate_dashboard_data import generate_equity_curve

    generate_equity_curve()


def get_last_processed_date(cfg: dict) -> date | None:
    """
    Last calendar day with paper-trading output on disk.

    Checks paper_equity.csv, run_daily daily logs (YYYY-MM-DD.jsonl),
    and positions_current.json (as_of). Returns None if nothing exists yet.
    """
    from datetime import timedelta

    candidates: list[date] = []

    if PAPER_EQUITY_PATH.exists():
        pe = pd.read_csv(PAPER_EQUITY_PATH, parse_dates=[0], index_col=0)
        if not pe.empty:
            candidates.append(pd.Timestamp(pe.index.max()).date())

    trades_dir = ROOT / "logs" / "trades"
    if trades_dir.exists():
        for path in trades_dir.glob("????-??-??.jsonl"):
            try:
                candidates.append(date.fromisoformat(path.stem))
            except ValueError:
                continue

    pos_path = ROOT / "logs" / "positions_current.json"
    if pos_path.exists():
        try:
            data = json.loads(pos_path.read_text(encoding="utf-8"))
            as_of = data.get("as_of")
            if as_of:
                candidates.append(date.fromisoformat(str(as_of)[:10]))
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    if candidates:
        return max(candidates)

    paper_start = cfg.get("data", {}).get("paper_start")
    if paper_start:
        return date.fromisoformat(str(paper_start)[:10]) - timedelta(days=1)
    return None


def iter_catch_up_dates(
    cfg: dict,
    *,
    through: date | None = None,
    force: bool = False,
) -> list[date]:
    """
    NYSE trading days to process after the last processed date, up to ``through``.

    Default ``through`` is yesterday. Skips days that already have a daily log
  unless ``force`` is True.
    """
    from datetime import timedelta

    import pandas_market_calendars as mcal

    through = through or (date.today() - timedelta(days=1))
    last = get_last_processed_date(cfg)

    paper_start_raw = cfg.get("data", {}).get("paper_start", "2026-01-02")
    paper_start = date.fromisoformat(str(paper_start_raw)[:10])

    if last is None:
        start = paper_start
    else:
        start = last + timedelta(days=1)

    if start > through:
        return []

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=str(start), end_date=str(through))
    if schedule.empty:
        return []

    trading_days = [ts.date() for ts in schedule.index]

    if force:
        return trading_days

    trades_dir = ROOT / "logs" / "trades"
    pending: list[date] = []
    for d in trading_days:
        log_path = trades_dir / f"{d}.jsonl"
        if log_path.exists() and log_path.stat().st_size > 0:
            continue
        pending.append(d)
    return pending


def daily_log_exists(target_date: date) -> bool:
    path = ROOT / "logs" / "trades" / f"{target_date}.jsonl"
    return path.exists() and path.stat().st_size > 0
