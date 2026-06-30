"""
Experiments router — compare all experiment runs with daily caching.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Request

from dashboard.api.schemas import ExperimentSummary

ROOT = Path(__file__).resolve().parent.parent.parent.parent

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@lru_cache(maxsize=1)
def _load_all_experiments(cache_date: date) -> list[dict]:
    """
    Scan all experiments/*/results.json and extract summary metrics.
    cache_date acts as a daily cache key — results refresh once per day.
    """
    experiments_dir = ROOT / "experiments"
    if not experiments_dir.exists():
        return []

    summaries = []
    for results_file in experiments_dir.glob("*/results.json"):
        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
            exp_id = results_file.parent.name
            mas_metrics = data.get("mas", {}).get("metrics", {})
            window_results = data.get("mas", {}).get("window_results", [])

            summaries.append({
                "experiment_id": exp_id,
                "timestamp": data.get("timestamp"),
                "total_return_pct": mas_metrics.get("total_return_pct"),
                "annualized_return_pct": mas_metrics.get("annualized_return_pct"),
                "sharpe_ratio": mas_metrics.get("sharpe_ratio"),
                "max_drawdown_pct": mas_metrics.get("max_drawdown_pct"),
                "calmar_ratio": mas_metrics.get("calmar_ratio"),
                "win_rate_pct": mas_metrics.get("win_rate_pct"),
                "n_trading_days": mas_metrics.get("n_trading_days"),
                "n_windows": len(window_results),
            })
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    summaries.sort(
        key=lambda x: x.get("sharpe_ratio") or -999,
        reverse=True,
    )
    return summaries


@router.get("/", response_model=list[ExperimentSummary])
def list_experiments(request: Request) -> list[dict]:
    return _load_all_experiments(date.today())
