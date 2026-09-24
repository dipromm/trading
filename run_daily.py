#!/usr/bin/env python3
"""
Daily paper trading pipeline — run post-market (~23:00 UTC).

Loads serialized models (NO .fit()), downloads T-1 data, runs inference,
logs decisions, updates positions, and generates XAI artifacts.

Usage:
    python run_daily.py
        # catch-up: all NYSE days from last processed through yesterday
    python run_daily.py --no-catch-up
        # only yesterday (typical cron)
    python run_daily.py --date 2026-06-27
        # single specific date

Cron example (VPS):
    0 23 * * 1-5 cd /app && python run_daily.py >> logs/cron.log 2>&1
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def is_market_open(target_date: date) -> bool:
    """Check if NYSE was open on target_date."""
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=str(target_date),
        end_date=str(target_date),
    )
    return len(schedule) > 0


def load_serialized_models() -> tuple:
    """Load pre-trained models from disk. Never calls .fit()."""
    models_dir = ROOT / "models"
    mat_path = models_dir / "matematico_v1.joblib"
    juez_path = models_dir / "juez_v1.joblib"

    if not mat_path.exists() or not juez_path.exists():
        raise FileNotFoundError(
            f"Model files not found in {models_dir}/. "
            "Run 'python scripts/setup_models.py' first."
        )

    matematico_model = joblib.load(mat_path)
    juez_model = joblib.load(juez_path)

    return matematico_model, juez_model


def download_latest_data(cfg: dict, target_date: date) -> tuple[dict, pd.DataFrame]:
    """Download T-1 data for all tickers + VIX, extending end_date to target_date."""
    import copy
    from mas.data.downloader import download_all
    from mas.data.features import compute_all_features
    from mas.data.regime import build_regime_features, download_vix

    # Extend data range to cover target_date so paper-trading inference uses current prices
    dl_cfg = copy.deepcopy(cfg)
    paper_end = str(target_date + timedelta(days=1))
    if paper_end > dl_cfg["data"]["end_date"]:
        dl_cfg["data"]["end_date"] = paper_end

    prices = download_all(dl_cfg, force_download=True)
    features = {t: compute_all_features(df, cfg) for t, df in prices.items()}
    vix = download_vix(cfg, force_download=True)
    regime_features = build_regime_features(prices, vix, cfg)

    return prices, features, regime_features


def run_inference(
    cfg: dict,
    matematico_model,
    juez_model,
    features: dict,
    regime_features: pd.DataFrame,
    target_date: date,
) -> list[dict]:
    """
    Run the full inference pipeline for a single date.
    Returns list of decision dicts for each ticker.
    """
    from mas.agents.matematico import FEATURE_COLUMNS
    from mas.agents.gestor_riesgos import GestorRiesgos
    from mas.agents.conspiranoico import Conspiranoico
    from mas.agents.cazador import Cazador

    gestor = GestorRiesgos.from_config(cfg)
    conspiranoico = Conspiranoico(cfg)
    cazador = Cazador(cfg)

    target_ts = pd.Timestamp(target_date)
    kelly_fraction = cfg["risk_manager"]["kelly_fraction"]

    # Determine veto/risk_scale from regime features
    con_cfg = cfg.get("conspiranoico", {})
    veto_active = False
    risk_scale = 1.0

    if regime_features is not None and not regime_features.empty:
        train_end = target_ts - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=3)
        train_regime = regime_features.loc[str(train_start.date()):str(train_end.date())]

        if len(train_regime) > 30:
            try:
                conspiranoico.fit(train_regime)
                today_regime = regime_features.loc[[str(target_date)]] if str(target_date) in regime_features.index.astype(str).tolist() else None

                if today_regime is not None and not today_regime.empty:
                    if conspiranoico.veto_mode == "soft":
                        scale = conspiranoico.predict_risk_scale(today_regime)
                        risk_scale = float(scale.iloc[0]) if not scale.empty else 1.0
                        veto_active = risk_scale < 1.0
                    else:
                        veto_raw = conspiranoico.predict(today_regime)
                        veto_active = int(veto_raw.iloc[0]) == 1 if not veto_raw.empty else False
            except Exception as exc:
                logger.warning("Conspiranoico failed: %s (defaulting to no veto)", exc)

    decisions = []
    tickers = list(features.keys())

    for ticker in tickers:
        feat_df = features[ticker]
        if feat_df.empty:
            continue

        last_row = feat_df.iloc[[-1]]
        if last_row.index[0].date() > target_date:
            continue

        missing_feats = [c for c in FEATURE_COLUMNS if c not in last_row.columns]
        if missing_feats:
            continue

        feat_clean = last_row[FEATURE_COLUMNS].dropna(axis=0, how="any")
        if feat_clean.empty:
            continue

        try:
            prob = matematico_model.predict_proba(feat_clean[FEATURE_COLUMNS])[:, 1]
            p = float(prob[0])
        except Exception:
            continue

        # Cazador alert
        try:
            caz_pred = cazador.predict_ticker(feat_df.tail(30), ticker)
            cazador_signal = "signal" if (caz_pred.iloc[-1] == 1 if not caz_pred.empty else False) else "silence"
        except Exception:
            cazador_signal = "silence"

        # Compute gain/loss ratio from recent returns
        if "return_1d" in feat_df.columns:
            recent_rets = feat_df["return_1d"].dropna().tail(252)
            b = gestor.compute_gain_loss_ratio(recent_rets) if len(recent_rets) > 20 else 1.0
        else:
            b = 1.0

        # Apply veto
        if veto_active:
            action = "HOLD"
            f_star = 0.0
        else:
            rho = kelly_fraction * risk_scale
            asset_class = "equity"  # simplified; full version uses ticker_classes
            f_star = gestor.compute_position_size(p, b, asset_class, kelly_fraction=rho)
            action = "BUY" if f_star > 0 else "HOLD"

        position_size_eur = f_star * cfg["backtester"]["initial_capital"]

        decisions.append({
            "date": str(target_date),
            "ticker": ticker,
            "action": action,
            "probability": round(p, 4),
            "kelly_fraction": round(f_star, 5),
            "kelly_inputs": {
                "p": round(p, 4),
                "b": round(b, 4),
                "rho": round(kelly_fraction * risk_scale, 4),
            },
            "agent_votes": {
                "matematico": round(p, 4),
                "analista": None,
                "cazador": cazador_signal,
                "conspiranoico": "active" if veto_active else "inactive",
            },
            "reason": "veto" if veto_active else ("signal" if f_star > 0 else "below_threshold"),
            "position_size_eur": round(position_size_eur, 2),
        })

    return decisions


def update_positions(decisions: list[dict], target_date: date, cfg: dict) -> None:
    """Update logs/positions_current.json based on today's decisions."""
    from mas.utils.paper_equity import (
        apply_decisions_with_cash,
        build_positions_state,
        get_cash_balance,
    )

    positions_path = ROOT / "logs" / "positions_current.json"

    current: dict = {}
    if positions_path.exists():
        try:
            current = json.loads(positions_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = {}

    positions = dict(current.get("positions") or {})
    cash = get_cash_balance(current, cfg)
    positions, cash = apply_decisions_with_cash(positions, cash, decisions)

    current_state = build_positions_state(positions, cash, target_date)

    positions_path.parent.mkdir(parents=True, exist_ok=True)
    positions_path.write_text(
        json.dumps(current_state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def generate_xai_artifacts(
    matematico_model,
    features: dict,
    cfg: dict,
) -> None:
    """Generate SHAP beeswarm and reliability diagram PNGs with dark theme."""
    from mas.agents.matematico import FEATURE_COLUMNS

    static_dir = ROOT / "dashboard" / "api" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    # --- SHAP Beeswarm ---
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample_dfs = []
        for ticker, df in list(features.items())[:10]:
            clean = df[FEATURE_COLUMNS].dropna().tail(50)
            if not clean.empty:
                sample_dfs.append(clean)

        if sample_dfs:
            sample = pd.concat(sample_dfs).head(200)
            base_estimators = matematico_model.calibrated_classifiers_
            first_base = base_estimators[0].estimator

            explainer = shap.TreeExplainer(first_base)
            shap_values = explainer.shap_values(sample)

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("#111111")
            ax.set_facecolor("#1a1a1a")

            shap.summary_plot(
                shap_values, sample,
                show=False, plot_size=(10, 8),
            )
            plt.tight_layout()
            plt.savefig(
                static_dir / "shap_beeswarm.png",
                dpi=150, facecolor="#111111", bbox_inches="tight",
            )
            plt.close()
            logger.info("SHAP beeswarm generated")
    except Exception as exc:
        logger.warning("Failed to generate SHAP beeswarm: %s", exc)

    # --- Reliability Diagram data (JSON for Recharts) ---
    try:
        reliability_data = {
            "calibration_curve": [],
            "perfect_calibration": [
                {"mean_predicted": i / 10, "fraction_positives": i / 10}
                for i in range(11)
            ],
        }
        (static_dir / "reliability_diagram.json").write_text(
            json.dumps(reliability_data, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to generate reliability diagram: %s", exc)


def update_registry(status: str, run_at: str) -> None:
    """Update registry.json with last_run_at and last_run_status."""
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.exists():
        return

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    data["active_model"]["last_run_at"] = run_at
    data["active_model"]["last_run_status"] = status
    registry_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_single_day(
    target_date: date,
    cfg: dict,
    matematico_model,
    juez_model,
    *,
    refresh_equity_curve: bool = False,
    generate_xai: bool = False,
) -> tuple[int, dict | None, dict | None]:
    """
    Run the pipeline for one trading day.

    Returns (exit_code, prices, features) — prices/features set on success.
    """
    if not is_market_open(target_date):
        logger.info("Market closed on %s — skipping", target_date)
        return 0, None, None

    try:
        logger.info("=== Processing %s ===", target_date)

        logger.info("Downloading T-1 data for %s...", target_date)
        prices, features, regime_features = download_latest_data(cfg, target_date)
        logger.info("  %d tickers loaded", len(prices))

        logger.info("Running inference pipeline...")
        decisions = run_inference(
            cfg, matematico_model, juez_model,
            features, regime_features, target_date,
        )
        logger.info("  %d decisions generated", len(decisions))

        buy_count = sum(1 for d in decisions if d["action"] == "BUY")
        hold_count = sum(1 for d in decisions if d["action"] == "HOLD")
        logger.info("  BUY: %d | HOLD: %d", buy_count, hold_count)

        logs_dir = ROOT / "logs" / "trades"
        logs_dir.mkdir(parents=True, exist_ok=True)

        daily_log = logs_dir / f"{target_date}.jsonl"
        with open(daily_log, "w", encoding="utf-8") as f:
            for d in decisions:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        logger.info("  Logged to %s", daily_log)

        update_positions(decisions, target_date, cfg)

        from mas.utils.paper_equity import (
            append_paper_equity_row,
            compute_portfolio_value,
            load_positions_state,
            refresh_dashboard_equity_curve,
        )
        pos_state = load_positions_state()
        portfolio_value = compute_portfolio_value(pos_state, prices, target_date, cfg)
        append_paper_equity_row(target_date, portfolio_value)

        if refresh_equity_curve:
            refresh_dashboard_equity_curve()

        if generate_xai:
            logger.info("Generating XAI artifacts...")
            generate_xai_artifacts(matematico_model, features, cfg)

        logger.info("Pipeline completed successfully for %s", target_date)
        return 0, prices, features

    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", target_date, exc)
        logger.error(traceback.format_exc())

        errors_dir = ROOT / "logs" / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)
        error_log = errors_dir / f"{target_date}.json"
        error_log.write_text(
            json.dumps({
                "date": str(target_date),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 1, None, None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Daily paper trading pipeline")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Process a single date (YYYY-MM-DD). Disables automatic catch-up.",
    )
    parser.add_argument(
        "--no-catch-up", action="store_true",
        help="Only process yesterday instead of filling all gaps since last run.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process days that already have a daily log file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    yesterday = date.today() - timedelta(days=1)

    from mas.utils.config_loader import load_config
    from mas.utils.paper_equity import (
        get_last_processed_date,
        iter_catch_up_dates,
        refresh_dashboard_equity_curve,
    )
    from mas.utils.reproducibility import set_all_seeds

    cfg = load_config(
        profile_path="profiles/exp1_menos_friccion.yaml",
        force_reload=True,
    )
    set_all_seeds(cfg["general"]["random_seed"])

    if args.date:
        dates_to_run = [date.fromisoformat(args.date)]
    elif args.no_catch_up:
        dates_to_run = [yesterday]
    else:
        last = get_last_processed_date(cfg)
        dates_to_run = iter_catch_up_dates(cfg, through=yesterday, force=args.force)
        if last and dates_to_run:
            logger.info(
                "Catch-up: last processed %s -> running %d day(s) through %s",
                last,
                len(dates_to_run),
                yesterday,
            )
        elif last and not dates_to_run:
            logger.info("Already up to date (last processed: %s)", last)

    if not dates_to_run:
        update_registry("up_to_date", run_timestamp)
        return 0

    if not args.date:
        dates_to_run = [d for d in dates_to_run if is_market_open(d)]
        if not dates_to_run:
            logger.info("No open market days to process in range — exiting cleanly")
            update_registry("market_closed", run_timestamp)
            return 0

    try:
        logger.info("Loading serialized models...")
        matematico_model, juez_model = load_serialized_models()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        update_registry("error", run_timestamp)
        return 1

    exit_code = 0
    last_features: dict | None = None
    processed = 0

    for target_date in dates_to_run:
        rc, _prices, features = run_single_day(
            target_date,
            cfg,
            matematico_model,
            juez_model,
            refresh_equity_curve=False,
            generate_xai=False,
        )
        if rc != 0:
            exit_code = rc
            update_registry("error", run_timestamp)
            break
        if features is not None:
            last_features = features
            processed += 1

    if processed > 0:
        logger.info("Refreshing dashboard equity curve (%d day(s) processed)...", processed)
        refresh_dashboard_equity_curve()
        if last_features is not None:
            logger.info("Generating XAI artifacts (latest day)...")
            generate_xai_artifacts(matematico_model, last_features, cfg)
        update_registry("success", run_timestamp)
    elif exit_code == 0:
        update_registry("up_to_date", run_timestamp)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
