#!/usr/bin/env python3
"""
Backfill paper_equity.csv from run_daily daily logs (YYYY-MM-DD.jsonl).

Reconstructs the paper-trading equity curve by replaying BUY/SELL decisions
with proper mark-to-market pricing.  Positions and cash are carried forward
day-by-day, starting from the last known state before the backfill window.

Usage:
    python -m scripts.backfill_paper_equity
    python -m scripts.backfill_paper_equity --from 2026-06-18 --to 2026-06-30
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill paper equity from daily logs")
    parser.add_argument("--from", dest="from_date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    from mas.utils.config_loader import load_config
    from mas.utils.paper_equity import (
        PAPER_EQUITY_PATH,
        append_paper_equity_row,
        apply_decisions_with_cash,
        build_positions_state,
        compute_portfolio_value,
        refresh_dashboard_equity_curve,
    )
    from mas.data.downloader import download_all

    cfg = load_config(profile_path="profiles/exp1_menos_friccion.yaml", force_reload=True)
    trades_dir = ROOT / "logs" / "trades"
    daily_logs = sorted(trades_dir.glob("????-??-??.jsonl"))

    if not daily_logs:
        logger.error("No daily logs found in %s", trades_dir)
        return 1

    all_dates = [date.fromisoformat(p.stem) for p in daily_logs]
    start = date.fromisoformat(args.from_date) if args.from_date else min(all_dates)
    end = date.fromisoformat(args.to_date) if args.to_date else max(all_dates)
    logs_by_date = {p.stem: p for p in daily_logs}

    # --- Seed state: cash from last paper_equity row before start, positions empty ---
    # (We can't reliably recover open positions from paper_equity alone;
    #  the loop will rebuild positions by replaying the trade logs.)
    initial_capital = float(cfg["backtester"]["initial_capital"])
    cash: float = initial_capital
    positions: dict = {}

    if PAPER_EQUITY_PATH.exists():
        import pandas as pd
        pe = pd.read_csv(PAPER_EQUITY_PATH, parse_dates=[0], index_col=0)
        pe.index = pd.to_datetime(pe.index).normalize()
        prior = pe[pe.index < pd.Timestamp(start)]
        if not prior.empty:
            cash = float(prior["equity"].iloc[-1])
            logger.info("Seed cash from paper_equity before %s: %.2f EUR", start, cash)
        else:
            logger.info("No prior paper_equity data; seeding with initial capital %.2f EUR", cash)

    # --- Download prices extended to cover the full backfill window ---
    dl_cfg = copy.deepcopy(cfg)
    dl_cfg["data"]["end_date"] = str(end + timedelta(days=1))
    logger.info("Downloading prices through %s...", dl_cfg["data"]["end_date"])
    prices = download_all(dl_cfg, force_download=True)
    logger.info("Prices loaded for %d tickers", len(prices))

    # --- Replay each day ---
    current = start
    n = 0
    last_state: dict | None = None
    while current <= end:
        ds = current.isoformat()
        log_path = logs_by_date.get(ds)
        if log_path:
            decisions = []
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        decisions.append(json.loads(line))

            positions, cash = apply_decisions_with_cash(positions, cash, decisions)
            state = build_positions_state(positions, cash, current)
            value = compute_portfolio_value(state, prices, current, cfg)
            append_paper_equity_row(current, value)
            last_state = state
            n += 1
            logger.info(
                "  %s: %.2f EUR  cash=%.2f  %d positions",
                ds, value, state["cash"], len(positions),
            )
        current += timedelta(days=1)

    if n == 0:
        logger.warning("No daily logs found in range %s -> %s", start, end)
        return 1

    # Persist final positions so run_daily can continue from here
    if last_state is not None:
        positions_path = ROOT / "logs" / "positions_current.json"
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        positions_path.write_text(
            json.dumps(last_state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("positions_current.json updated (as_of=%s)", last_state.get("as_of"))

    logger.info("Backfilled %d trading days. Regenerating equity_curve.csv...", n)
    refresh_dashboard_equity_curve()
    logger.info("Done. Run 'python -m scripts.generate_dashboard_data' to update metrics too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
