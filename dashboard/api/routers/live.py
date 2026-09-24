"""
Live data router — system status, positions, council verdict, trade log.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query, Request

from dashboard.api.schemas import (
    CouncilVerdict,
    MarketCloseResponse,
    PaperTradingLog,
    PositionsResponse,
    RegistryStatus,
    StatusMessage,
)

ROOT = Path(__file__).resolve().parent.parent.parent.parent

router = APIRouter(prefix="/api/live", tags=["live"])


def _paper_trading_started() -> bool:
    """Check if paper trading has generated any logs."""
    trades_dir = ROOT / "logs" / "trades"
    if not trades_dir.exists():
        return False
    paper_log = trades_dir / "paper.jsonl"
    if paper_log.exists() and paper_log.stat().st_size > 0:
        return True
    daily_logs = sorted(trades_dir.glob("2*.jsonl"))
    return len(daily_logs) > 0


def _position_value(pos: dict) -> float:
    """Best-effort EUR value for a single open position."""
    if pos.get("current_value") is not None:
        return float(pos["current_value"])
    if pos.get("allocated_eur") is not None:
        return float(pos["allocated_eur"])
    qty = float(pos.get("quantity") or 0)
    price = float(pos.get("entry_price") or 0)
    return qty * price


def _normalize_positions_payload(raw: dict) -> dict:
    """
    Normalize positions_current.json into the shape expected by the frontend.

    run_daily.py stores positions as a dict keyed by ticker; the dashboard
    expects a list of PositionEntry objects.
    """
    positions_raw = raw.get("positions", [])
    entries: list[dict] = []

    if isinstance(positions_raw, dict):
        for ticker, pos in positions_raw.items():
            pos = dict(pos)
            entries.append({
                "ticker": ticker,
                "quantity": pos.get("quantity", 0),
                "entry_price": pos.get("entry_price", 0),
                "entry_date": pos.get("entry_date", ""),
                "current_value": _position_value(pos),
                "unrealized_pnl": pos.get("unrealized_pnl"),
            })
    elif isinstance(positions_raw, list):
        for pos in positions_raw:
            pos = dict(pos)
            if "ticker" not in pos:
                continue
            entries.append({
                "ticker": pos["ticker"],
                "quantity": pos.get("quantity", 0),
                "entry_price": pos.get("entry_price", 0),
                "entry_date": pos.get("entry_date", ""),
                "current_value": _position_value(pos),
                "unrealized_pnl": pos.get("unrealized_pnl"),
            })

    allocated = sum(e["current_value"] or 0 for e in entries)
    cash = float(raw.get("cash") or 0)

    # run_daily may report negative cash when allocations exceed initial_capital
    if cash < 0 and allocated > 0:
        cash = 0.0

    total_value = raw.get("total_value")
    if total_value is None:
        total_value = allocated + cash

    return {
        "positions": entries,
        "cash": cash,
        "total_value": float(total_value),
        "as_of": raw.get("as_of"),
    }


@router.get("/market-close", response_model=MarketCloseResponse)
def get_market_close(request: Request) -> MarketCloseResponse:
    """Last NYSE market close date and pipeline execution time."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        today = pd.Timestamp.now(tz="US/Eastern").normalize()
        schedule = nyse.schedule(
            start_date=today - pd.Timedelta(days=10),
            end_date=today,
        )
        if schedule.empty:
            last_close_str = "Unknown"
        else:
            last_close = schedule.index[-1]
            if last_close > today:
                last_close = schedule.index[-2] if len(schedule) > 1 else schedule.index[-1]
            last_close_str = last_close.strftime("%A, %b %d, %Y")
    except Exception:
        last_close_str = "Unknown (pandas_market_calendars unavailable)"

    pipeline_ran_at = None
    registry_path = ROOT / "models" / "registry.json"
    if registry_path.exists():
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        last_run = data.get("active_model", {}).get("last_run_at")
        if last_run:
            try:
                dt = datetime.fromisoformat(last_run)
                pipeline_ran_at = dt.strftime("%b %d at %H:%M UTC")
            except (ValueError, TypeError):
                pipeline_ran_at = str(last_run)

    return MarketCloseResponse(
        last_market_close=last_close_str,
        pipeline_ran_at=pipeline_ran_at,
    )


@router.get("/status", response_model=RegistryStatus | StatusMessage)
def get_status(request: Request):
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.exists():
        return StatusMessage(status="not_configured")

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    model_info = data.get("active_model", {})
    return RegistryStatus(**model_info)


@router.get("/positions")
def get_positions(request: Request):
    positions_path = ROOT / "logs" / "positions_current.json"
    if not positions_path.exists():
        if not _paper_trading_started():
            return StatusMessage(status="not_started")
        return PositionsResponse(positions=[], total_value=0, cash=0)

    data = json.loads(positions_path.read_text(encoding="utf-8"))
    return _normalize_positions_payload(data)


@router.get("/council-verdict")
def get_council_verdict(request: Request):
    if not _paper_trading_started():
        return StatusMessage(status="not_started")

    # 1. Look for a date-named daily log (YYYY-MM-DD.jsonl) written by run_daily.py
    trades_dir = ROOT / "logs" / "trades"
    daily_logs = sorted(trades_dir.glob("2*.jsonl"), reverse=True)
    if daily_logs:
        latest_log = daily_logs[0]
        decisions = []
        with open(latest_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    decisions.append(json.loads(line))
        log_date = latest_log.stem
        return {"date": log_date, "decisions": decisions, "veto_active": False}

    # 2. Group paper.jsonl trades by date and return the most recent day
    paper_log = trades_dir / "paper.jsonl"
    if paper_log.exists():
        by_date: dict[str, list] = {}
        with open(paper_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                d = record.get("date", "")
                if d:
                    by_date.setdefault(d, []).append(record)
        if by_date:
            latest_date = sorted(by_date.keys())[-1]
            return {
                "date": latest_date,
                "decisions": by_date[latest_date],
                "veto_active": False,
            }

    return StatusMessage(status="not_started")


@router.get("/trades")
def get_trades(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    if not _paper_trading_started():
        return StatusMessage(status="not_started")

    trades_dir = ROOT / "logs" / "trades"
    all_trades: list[dict] = []

    paper_log = trades_dir / "paper.jsonl"
    if paper_log.exists():
        with open(paper_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_trades.append(json.loads(line))

    daily_logs = sorted(trades_dir.glob("2*.jsonl"))
    for log_file in daily_logs:
        if log_file.name == "paper.jsonl":
            continue
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_trades.append(json.loads(line))

    all_trades.sort(key=lambda t: t.get("date", ""), reverse=True)
    total = len(all_trades)
    paginated = all_trades[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": paginated,
    }
