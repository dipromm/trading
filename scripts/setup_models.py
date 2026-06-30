#!/usr/bin/env python3
"""
Setup script — Serialize models and rebuild equity_curve.csv.

Run once after cloning the repo (on VPS or local) to produce the .joblib
model files and the multi-period equity curve needed by the dashboard API.

Usage:
    python scripts/setup_models.py
    python scripts/setup_models.py --profile profiles/exp1_menos_friccion.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

CANONICAL_EXPERIMENT = (
    "fase6_mat_cazador_conspiranoico_exp1_menos_friccion_20260622_035259"
)
DEFAULT_PROFILE = "profiles/exp1_menos_friccion.yaml"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def train_and_serialize_models(cfg: dict) -> dict[str, str]:
    """
    Train the canonical models using the full pre-holdout data and serialize.

    Returns dict of {model_name: sha256_hash}.
    """
    from agents.matematico import Matematico
    from agents.cazador import Cazador
    from agents.gestor_riesgos import GestorRiesgos
    from agents.conspiranoico import Conspiranoico
    from judge.judge_v1 import JuezV1
    from data.downloader import download_all
    from data.features import compute_all_features
    from data.regime import build_regime_features, download_vix
    from backtester.walk_forward import WalkForwardValidator

    logger.info("Downloading prices and computing features...")
    prices = download_all(cfg)
    features = {t: compute_all_features(df, cfg) for t, df in prices.items()}
    logger.info("  %d tickers loaded", len(prices))

    vix = download_vix(cfg)
    regime_features = build_regime_features(prices, vix, cfg)

    matematico = Matematico(cfg)
    gestor = GestorRiesgos.from_config(cfg)
    juez = JuezV1(cfg)
    cazador = Cazador(cfg)
    conspiranoico = Conspiranoico(cfg)

    validator = WalkForwardValidator(cfg, include_holdout=False)
    logger.info("Running walk-forward (%d windows) to train models...", len(validator.windows))

    validator.run(
        agents={
            "matematico": matematico,
            "cazador": cazador,
            "conspiranoico": conspiranoico,
        },
        judge=juez,
        gestor=gestor,
        prices=prices,
        features=features,
        regime_features=regime_features,
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    mat_path = models_dir / "matematico_v1.joblib"
    juez_path = models_dir / "juez_v1.joblib"

    joblib.dump(matematico.model, mat_path)
    logger.info("Serialized matematico -> %s", mat_path)

    joblib.dump(juez.model, juez_path)
    logger.info("Serialized juez -> %s", juez_path)

    hashes = {
        "matematico": sha256_file(mat_path),
        "juez": sha256_file(juez_path),
    }

    registry_path = models_dir / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["active_model"]["model_hash_sha256"] = hashes
    registry["active_model"]["trained_at"] = datetime.now().isoformat(timespec="seconds")
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Registry updated with SHA-256 hashes")

    return hashes


def rebuild_equity_curve(cfg: dict) -> Path:
    """
    Rebuild logs/dashboard/equity_curve.csv with period, buyhold, and SMA columns.

    Sources:
      - Walk-forward: experiments/<canonical>/equity_curve.csv
      - Holdout: logs/dashboard/holdout_cache/ (if exists)
      - Paper trading: logs/trades/paper.jsonl (if exists)
    """
    from baselines.buy_and_hold import run as bh_run
    from baselines.sma_crossover import run as sma_run
    from data.downloader import download_all

    initial_capital = cfg["backtester"]["initial_capital"]
    holdout_start = cfg["data"]["holdout_start"]
    data_end = cfg["data"]["end_date"]

    # --- Walk-forward equity (from canonical experiment) ---
    exp_dir = ROOT / "experiments" / CANONICAL_EXPERIMENT
    wf_csv = exp_dir / "equity_curve.csv"
    if not wf_csv.exists():
        raise FileNotFoundError(
            f"Canonical equity_curve.csv not found: {wf_csv}\n"
            "Run the canonical experiment first with: python run.py --profile profiles/exp1_menos_friccion.yaml"
        )

    wf_raw = pd.read_csv(wf_csv)
    date_col = "Date" if "Date" in wf_raw.columns else "date"
    wf_raw[date_col] = pd.to_datetime(wf_raw[date_col])
    wf_df = wf_raw.set_index(date_col)
    wf_df = wf_df.rename(columns={"equity": "portfolio_value"})
    wf_df = wf_df[wf_df.index < pd.Timestamp(holdout_start)]
    wf_df["period"] = "walkforward"

    # --- Holdout equity (from holdout cache or re-run) ---
    holdout_metrics_path = ROOT / "logs" / "dashboard" / "metrics.json"
    holdout_eq_path = ROOT / "logs" / "dashboard" / "equity_curve.csv"

    if holdout_eq_path.exists():
        ho_csv = pd.read_csv(holdout_eq_path)
        ho_date_col = "Date" if "Date" in ho_csv.columns else "date"
        ho_csv[ho_date_col] = pd.to_datetime(ho_csv[ho_date_col])
        ho_raw = ho_csv.set_index(ho_date_col)
        ho_raw = ho_raw.rename(columns={"equity": "portfolio_value"})
        ho_df = ho_raw[ho_raw.index >= pd.Timestamp(holdout_start)]

        if not wf_df.empty and not ho_df.empty:
            wf_final = wf_df["portfolio_value"].iloc[-1]
            ho_first = ho_df["portfolio_value"].iloc[0]
            if ho_first != 0:
                scale = wf_final / ho_first
                ho_df["portfolio_value"] = ho_df["portfolio_value"] * scale

        ho_df["period"] = "holdout"
    else:
        ho_df = pd.DataFrame(columns=["portfolio_value", "period"])

    # --- Paper trading equity (from paper.jsonl if exists) ---
    paper_log = ROOT / "logs" / "trades" / "paper.jsonl"
    if paper_log.exists():
        paper_trades = []
        with open(paper_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    paper_trades.append(json.loads(line))

        if paper_trades:
            paper_dates = sorted(set(t.get("date", t.get("entry_date", "")) for t in paper_trades))
            base_val = ho_df["portfolio_value"].iloc[-1] if not ho_df.empty else initial_capital
            paper_df = pd.DataFrame({
                "Date": pd.to_datetime(paper_dates),
                "portfolio_value": base_val,
                "period": "paper_trading",
            }).set_index("Date")
        else:
            paper_df = pd.DataFrame(columns=["portfolio_value", "period"])
    else:
        paper_df = pd.DataFrame(columns=["portfolio_value", "period"])

    # --- Buy & Hold and SMA baselines across full period ---
    prices = download_all(cfg)

    full_equity = pd.concat([wf_df, ho_df, paper_df])
    full_equity = full_equity[~full_equity.index.duplicated(keep="first")]
    full_equity = full_equity.sort_index()

    start_date = str(full_equity.index[0].date()) if not full_equity.empty else "2021-01-01"
    end_date_str = str(full_equity.index[-1].date()) if not full_equity.empty else data_end

    try:
        bh_eq, _ = bh_run(prices, start_date=start_date, end_date=end_date_str, config=cfg)
        if not bh_eq.empty:
            bh_scale = initial_capital / bh_eq.iloc[0] if bh_eq.iloc[0] != 0 else 1.0
            bh_aligned = (bh_eq * bh_scale).reindex(full_equity.index, method="ffill")
            full_equity["buyhold_value"] = bh_aligned.values
        else:
            full_equity["buyhold_value"] = np.nan
    except Exception as exc:
        logger.warning("Could not compute buy & hold baseline: %s", exc)
        full_equity["buyhold_value"] = np.nan

    try:
        sma_eq, _ = sma_run(prices, start_date=start_date, end_date=end_date_str, config=cfg)
        if not sma_eq.empty:
            sma_scale = initial_capital / sma_eq.iloc[0] if sma_eq.iloc[0] != 0 else 1.0
            sma_aligned = (sma_eq * sma_scale).reindex(full_equity.index, method="ffill")
            full_equity["sma_value"] = sma_aligned.values
        else:
            full_equity["sma_value"] = np.nan
    except Exception as exc:
        logger.warning("Could not compute SMA crossover baseline: %s", exc)
        full_equity["sma_value"] = np.nan

    # --- Write output ---
    out_dir = ROOT / "logs" / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "equity_curve.csv"

    output = full_equity[["portfolio_value", "buyhold_value", "sma_value", "period"]]
    output.index.name = "date"
    output.to_csv(out_path)

    logger.info(
        "Equity curve rebuilt: %d rows (%d wf, %d holdout, %d paper) -> %s",
        len(output),
        len(wf_df),
        len(ho_df),
        len(paper_df),
        out_path,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup models and rebuild equity curve")
    parser.add_argument(
        "--profile", default=DEFAULT_PROFILE,
        help="Profile YAML to use for training",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training (only rebuild equity curve)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from utils.config_loader import load_config
    from utils.reproducibility import set_all_seeds

    cfg = load_config(profile_path=args.profile, force_reload=True)
    set_all_seeds(cfg["general"]["random_seed"])

    if not args.skip_training:
        logger.info("=== Phase 1: Train and serialize models ===")
        hashes = train_and_serialize_models(cfg)
        logger.info("Model hashes: %s", hashes)
    else:
        logger.info("Skipping model training (--skip-training)")

    logger.info("=== Phase 2: Rebuild equity curve ===")
    out_path = rebuild_equity_curve(cfg)
    logger.info("Done. Output: %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
