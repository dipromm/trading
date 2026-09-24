"""
Generate unified dashboard data files from raw backtester outputs.

Reads the MAS equity curve, paper trading equity, and (if cached price data
is available) computes Buy & Hold and SMA Crossover baselines. Normalizes
all series to base 100 and writes the result to logs/dashboard/equity_curve.csv
in the format expected by the FastAPI EquityCurvePoint schema.

Also computes extended metrics (Sortino, Profit Factor) and writes per-period
metrics JSON files.

Usage:
    python -m scripts.generate_dashboard_data
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mas.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DASHBOARD_DIR = ROOT / "logs" / "dashboard"
CANONICAL_EXPERIMENT = (
    "fase6_mat_cazador_conspiranoico_exp1_menos_friccion_20260622_035259"
)


def _load_mas_equity() -> pd.DataFrame:
    """Load the MAS walk-forward equity curve from the canonical experiment."""
    exp_path = ROOT / "experiments" / CANONICAL_EXPERIMENT / "equity_curve.csv"
    fallback_path = DASHBOARD_DIR / "equity_curve.csv"

    csv_path = exp_path if exp_path.exists() else fallback_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MAS equity curve not found at {exp_path} or {fallback_path}"
        )

    df = pd.read_csv(csv_path)
    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", "equity": "portfolio_value"})
    if "date" in df.columns and "date" != df.index.name:
        df = df.set_index("date")
    df = df.sort_index()
    return df


def _load_paper_equity() -> pd.DataFrame | None:
    """Load paper trading equity if available."""
    csv_path = DASHBOARD_DIR / "paper_equity.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    df.index.name = "date"
    df = df.rename(columns={"equity": "portfolio_value"})
    df = df.sort_index()
    return df


def _try_load_baselines(
    dates: pd.DatetimeIndex,
    config: dict,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Attempt to compute baselines from cached price data.
    Returns (buyhold_series, sma_series) or (None, None) if data unavailable.
    """
    cache_dir = Path(config["data"]["cache_dir"])
    if not cache_dir.exists() or not any(cache_dir.glob("*.parquet")):
        logger.warning(
            "No cached price data found in %s. Baselines will be null. "
            "Run 'python -m mas.data.downloader' first to enable baseline curves.",
            cache_dir,
        )
        return None, None

    try:
        from mas.data.downloader import load_tickers, download_ticker
        from mas.baselines.buy_and_hold import run as run_bh
        from mas.baselines.sma_crossover import run as run_sma

        tickers = load_tickers(config)
        start = str(dates.min().date())
        end = str(dates.max().date())

        prices: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = download_ticker(
                ticker, start, end, cache_dir, force_download=False,
            )
            if df is not None and not df.empty:
                prices[ticker] = df

        if not prices:
            logger.warning("No price data loaded. Baselines will be null.")
            return None, None

        bh_equity, _ = run_bh(prices, start_date=start, end_date=end, config=config)
        sma_equity, _ = run_sma(prices, start_date=start, end_date=end, config=config)

        bh_series = bh_equity.reindex(dates, method="ffill") if not bh_equity.empty else None
        sma_series = sma_equity.reindex(dates, method="ffill") if not sma_equity.empty else None

        return bh_series, sma_series

    except Exception as exc:
        logger.warning("Failed to compute baselines: %s. They will be null.", exc)
        return None, None


def _normalize_base100(series: pd.Series) -> pd.Series:
    """Normalize a series so the first value equals 100."""
    first = series.iloc[0]
    if first == 0 or pd.isna(first):
        return series
    return series / first * 100


def _compute_sortino(daily_returns: pd.Series, annual_factor: float = 252) -> float:
    """Sortino ratio using downside deviation."""
    excess = daily_returns
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(annual_factor))


def _compute_profit_factor(daily_returns: pd.Series) -> float:
    """Gross profits / gross losses."""
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _compute_max_drawdown(equity: pd.Series) -> float:
    """Max drawdown as a negative percentage."""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100)


def _compute_metrics(equity: pd.Series) -> dict:
    """Compute full metrics dict from an equity series."""
    daily_returns = equity.pct_change().dropna()
    n_days = len(daily_returns)

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    ann_return = ((equity.iloc[-1] / equity.iloc[0]) ** (252 / max(n_days, 1)) - 1) * 100

    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    max_dd = _compute_max_drawdown(equity)
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

    positive_days = (daily_returns > 0).sum()
    win_rate = float(positive_days / n_days * 100) if n_days > 0 else 0.0

    return {
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(ann_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar_ratio": round(calmar, 3),
        "win_rate_pct": round(win_rate, 2),
        "n_trading_days": n_days,
        "sortino_ratio": round(_compute_sortino(daily_returns), 3),
        "profit_factor": round(_compute_profit_factor(daily_returns), 3),
    }


def generate_equity_curve() -> None:
    """Generate the unified equity_curve.csv for the dashboard."""
    config = load_config()
    holdout_start = config["data"]["holdout_start"]
    paper_start = config["data"].get("paper_start", "2026-01-02")

    mas_df = _load_mas_equity()
    paper_df = _load_paper_equity()

    bh_series, sma_series = _try_load_baselines(mas_df.index, config)

    # Normalize MAS to base 100
    mas_norm = _normalize_base100(mas_df["portfolio_value"])

    # Normalize baselines to same base 100
    bh_norm = _normalize_base100(bh_series) if bh_series is not None else None
    sma_norm = _normalize_base100(sma_series) if sma_series is not None else None

    # Build output DataFrame
    records: list[dict] = []
    for dt in mas_norm.index:
        date_str = dt.strftime("%Y-%m-%d")
        period = "holdout" if date_str >= holdout_start else "walkforward"
        records.append({
            "date": date_str,
            "portfolio_value": round(mas_norm[dt], 4),
            "buyhold_value": round(bh_norm[dt], 4) if bh_norm is not None and dt in bh_norm.index else None,
            "sma_value": round(sma_norm[dt], 4) if sma_norm is not None and dt in sma_norm.index else None,
            "period": period,
        })

    # Append paper trading (separate base, normalized to continue from last MAS value)
    if paper_df is not None and not paper_df.empty:
        last_mas_value = mas_norm.iloc[-1]
        paper_equity = paper_df["portfolio_value"]
        paper_scale = last_mas_value / paper_equity.iloc[0]
        paper_scaled = paper_equity * paper_scale

        for dt in paper_scaled.index:
            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "portfolio_value": round(paper_scaled[dt], 4),
                "buyhold_value": None,
                "sma_value": None,
                "period": "paper_trading",
            })

    out_df = pd.DataFrame(records)
    out_path = DASHBOARD_DIR / "equity_curve.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), out_path)


def generate_metrics() -> None:
    """
    Generate per-period metrics JSON files.
    Reads from canonical experiment results and holdout report.
    """
    results_path = ROOT / "experiments" / CANONICAL_EXPERIMENT / "results.json"
    holdout_path = ROOT / "reports" / "holdout_2025_exp1_menos_friccion.json"

    if not results_path.exists():
        logger.error("Canonical results not found: %s", results_path)
        return

    results = json.loads(results_path.read_text(encoding="utf-8"))
    mas_wf = results["mas"]["metrics"]
    bh_wf = results.get("buy_and_hold", {})
    sma_wf = results.get("sma_crossover", {})

    # Compute Sortino and Profit Factor from equity curve if available
    try:
        mas_df = _load_mas_equity()
        daily_returns = mas_df["portfolio_value"].pct_change().dropna()
        mas_wf["sortino_ratio"] = round(_compute_sortino(daily_returns), 3)
        mas_wf["profit_factor"] = round(_compute_profit_factor(daily_returns), 3)
    except Exception:
        mas_wf.setdefault("sortino_ratio", None)
        mas_wf.setdefault("profit_factor", None)

    # Walk-forward metrics
    wf_metrics = {
        "mas": mas_wf,
        "buy_and_hold": bh_wf,
        "sma_crossover": sma_wf,
    }
    wf_path = DASHBOARD_DIR / "metrics_walkforward.json"
    wf_path.write_text(json.dumps(wf_metrics, indent=2), encoding="utf-8")
    logger.info("Wrote walk-forward metrics to %s", wf_path)

    # Holdout metrics
    if holdout_path.exists():
        holdout_data = json.loads(holdout_path.read_text(encoding="utf-8"))
        holdout_mas = holdout_data.get("holdout_2025", {}).get("mas_metrics", {})
        holdout_bh = holdout_data.get("holdout_2025", {}).get("buy_and_hold", {})

        holdout_metrics = {
            "mas": holdout_mas,
            "buy_and_hold": holdout_bh,
            "sma_crossover": {},
        }
        holdout_path_out = DASHBOARD_DIR / "metrics_holdout.json"
        holdout_path_out.write_text(json.dumps(holdout_metrics, indent=2), encoding="utf-8")
        logger.info("Wrote holdout metrics to %s", holdout_path_out)

    # Update the main metrics.json to include new fields
    main_metrics = DASHBOARD_DIR / "metrics.json"
    main_metrics.write_text(json.dumps(mas_wf, indent=2), encoding="utf-8")
    logger.info("Updated main metrics.json")


if __name__ == "__main__":
    generate_equity_curve()
    generate_metrics()
    logger.info("Dashboard data generation complete.")
