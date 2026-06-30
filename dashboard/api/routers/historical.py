"""
Historical data router — equity curve, metrics, walk-forward windows, XAI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import APIRouter, Query, Request
from fastapi.responses import RedirectResponse

from dashboard.api.schemas import (
    BaselineMetrics,
    EquityCurvePoint,
    MetricsResponse,
    PeriodMetricsResponse,
    StatusMessage,
    WalkForwardWindow,
)

ROOT = Path(__file__).resolve().parent.parent.parent.parent

CANONICAL_EXPERIMENT = (
    "fase6_mat_cazador_conspiranoico_exp1_menos_friccion_20260622_035259"
)

router = APIRouter(prefix="/api/historical", tags=["historical"])


@router.get("/equity-curve", response_model=list[EquityCurvePoint])
def get_equity_curve(request: Request) -> list[dict]:
    csv_path = ROOT / "logs" / "dashboard" / "equity_curve.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path, parse_dates=["date"])
    records = []
    for _, row in df.iterrows():
        records.append({
            "date": row["date"].date() if hasattr(row["date"], "date") else row["date"],
            "portfolio_value": row["portfolio_value"],
            "buyhold_value": row.get("buyhold_value"),
            "sma_value": row.get("sma_value"),
            "period": row.get("period", "walkforward"),
        })
    return records


@router.get("/metrics")
def get_metrics(
    request: Request,
    period: Optional[str] = Query(default=None, description="walkforward | holdout"),
) -> PeriodMetricsResponse | MetricsResponse:
    """
    Returns metrics. If `period` is specified, returns that period's metrics
    with baselines. Otherwise returns overall MAS metrics.
    """
    dashboard_dir = ROOT / "logs" / "dashboard"

    if period == "walkforward":
        path = dashboard_dir / "metrics_walkforward.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return PeriodMetricsResponse(
                mas=MetricsResponse(**data.get("mas", {})),
                buy_and_hold=BaselineMetrics(**data.get("buy_and_hold", {})),
                sma_crossover=BaselineMetrics(**data.get("sma_crossover", {})),
            )

    if period == "holdout":
        path = dashboard_dir / "metrics_holdout.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return PeriodMetricsResponse(
                mas=MetricsResponse(**data.get("mas", {})),
                buy_and_hold=BaselineMetrics(**data.get("buy_and_hold", {})),
                sma_crossover=BaselineMetrics(**data.get("sma_crossover", {})),
            )

    # Fallback: read canonical experiment results
    results_path = ROOT / "experiments" / CANONICAL_EXPERIMENT / "results.json"
    if results_path.exists():
        exp = json.loads(results_path.read_text(encoding="utf-8"))
        mas_metrics = exp.get("mas", {}).get("metrics", {})
        if period is not None:
            return PeriodMetricsResponse(
                mas=MetricsResponse(**mas_metrics),
                buy_and_hold=BaselineMetrics(**exp.get("buy_and_hold", {})),
                sma_crossover=BaselineMetrics(**exp.get("sma_crossover", {})),
            )
        return MetricsResponse(**mas_metrics)

    return MetricsResponse(
        total_return_pct=0, annualized_return_pct=0, sharpe_ratio=0,
        max_drawdown_pct=0, calmar_ratio=0, win_rate_pct=0, n_trading_days=0,
    )


@router.get("/walkforward-windows", response_model=list[WalkForwardWindow])
def get_walkforward_windows(request: Request) -> list[dict]:
    results_path = (
        ROOT / "experiments" / CANONICAL_EXPERIMENT / "results.json"
    )
    if not results_path.exists():
        return []

    data = json.loads(results_path.read_text(encoding="utf-8"))
    windows = data.get("mas", {}).get("window_results", [])

    return [
        {
            "iteration": w.get("iteration", i + 1),
            "val_start": w.get("val_start", ""),
            "val_end": w.get("val_end", ""),
            "total_return_pct": w.get("total_return_pct", 0),
            "sharpe_ratio": w.get("sharpe_ratio", 0),
            "max_drawdown_pct": w.get("max_drawdown_pct", 0),
            "n_tickers": w.get("n_tickers", 0),
        }
        for i, w in enumerate(windows)
    ]


@router.get("/shap-beeswarm")
def get_shap_beeswarm(request: Request):
    png_path = Path(__file__).parent.parent / "static" / "shap_beeswarm.png"
    if not png_path.exists():
        return StatusMessage(status="not_generated")
    return RedirectResponse(url="/static/shap_beeswarm.png")


@router.get("/reliability-diagram")
def get_reliability_diagram(request: Request) -> dict:
    """
    Returns calibration curve data as JSON for Recharts rendering.
    Falls back to the canonical experiment's calibration data if available.
    """
    json_path = Path(__file__).parent.parent / "static" / "reliability_diagram.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))

    results_path = (
        ROOT / "experiments" / CANONICAL_EXPERIMENT / "results.json"
    )
    if results_path.exists():
        data = json.loads(results_path.read_text(encoding="utf-8"))
        calibration = data.get("mas", {}).get("calibration", {})
        if calibration:
            return calibration

    return {
        "status": "not_generated",
        "calibration_curve": [],
        "perfect_calibration": [],
    }
