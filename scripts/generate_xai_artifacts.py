#!/usr/bin/env python3
"""
Standalone XAI artifacts generator — runs independently of market-open check.

Generates:
  - dashboard/api/static/shap_beeswarm.png   (SHAP feature importance plot)
  - dashboard/api/static/reliability_diagram.json (calibration curve data)

Usage:
    python scripts/generate_xai_artifacts.py

Run this once after setup_models.py to populate the Lab page visualisations.
Subsequent runs are handled automatically by run_daily.py on trading days.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def generate_reliability_diagram(static_dir: Path) -> None:
    """
    Build calibration curve from paper.jsonl trade history.

    For each closed BUY trade we check whether the ticker's subsequent trade
    was at a higher price (positive outcome = 1) or lower (0).  We then bin
    the Matematico probability predictions and compute the empirical fraction
    of positives per bin — the classic reliability / calibration diagram.

    Falls back to an empty-curve stub when insufficient trade history exists.
    """
    import numpy as np

    trades_dir = ROOT / "logs" / "trades"
    paper_log = trades_dir / "paper.jsonl"

    samples: list[tuple[float, int]] = []  # (predicted_prob, outcome)

    if paper_log.exists():
        raw: list[dict] = []
        with open(paper_log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Group by ticker to compute outcomes
        from collections import defaultdict
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        for rec in raw:
            t = rec.get("ticker")
            if t:
                by_ticker[t].append(rec)

        for ticker, records in by_ticker.items():
            records.sort(key=lambda r: r.get("date", ""))
            for i, rec in enumerate(records[:-1]):
                if rec.get("action") != "BUY":
                    continue
                mat_prob = rec.get("agent_votes", {}).get("matematico")
                if mat_prob is None:
                    continue
                # Outcome: did the price increase by the next trade for this ticker?
                next_price = records[i + 1].get("price")
                curr_price = rec.get("price")
                if next_price is None or curr_price is None or curr_price == 0:
                    continue
                outcome = 1 if next_price > curr_price else 0
                try:
                    samples.append((float(mat_prob), outcome))
                except (TypeError, ValueError):
                    continue

    calibration_curve: list[dict] = []
    if len(samples) >= 20:
        probs = np.array([s[0] for s in samples])
        outcomes = np.array([s[1] for s in samples])
        n_bins = 5
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() < 3:
                continue
            calibration_curve.append(
                {
                    "mean_predicted": float(probs[mask].mean()),
                    "fraction_of_positives": float(outcomes[mask].mean()),
                    "n_samples": int(mask.sum()),
                }
            )
        logger.info(
            "Calibration curve: %d bins from %d BUY trades", len(calibration_curve), len(samples)
        )
    else:
        logger.info(
            "Only %d BUY trades available — calibration curve will populate "
            "after more paper-trading sessions.",
            len(samples),
        )

    perfect_calibration = [
        {"mean_predicted": round(i / 10, 1), "fraction_of_positives": round(i / 10, 1)}
        for i in range(11)
    ]

    payload = {
        "calibration_curve": calibration_curve,
        "perfect_calibration": perfect_calibration,
    }
    out_path = static_dir / "reliability_diagram.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Reliability diagram written to %s", out_path)


def generate_shap(static_dir: Path, matematico_model, features: dict, cfg: dict) -> None:
    """Generate SHAP beeswarm PNG using dark theme."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from agents.matematico import FEATURE_COLUMNS

        sample_dfs = []
        for ticker, df in list(features.items())[:15]:
            clean = df[FEATURE_COLUMNS].dropna().tail(60)
            if not clean.empty:
                sample_dfs.append(clean)

        if not sample_dfs:
            logger.warning("No feature data available for SHAP — skipping.")
            return

        import pandas as pd
        sample = pd.concat(sample_dfs).head(300)

        base_estimators = matematico_model.calibrated_classifiers_
        first_base = base_estimators[0].estimator

        explainer = shap.TreeExplainer(first_base)
        shap_values = explainer.shap_values(sample)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#1a1a1a")

        shap.summary_plot(shap_values, sample, show=False, plot_size=(10, 8))
        plt.tight_layout()

        out_path = static_dir / "shap_beeswarm.png"
        plt.savefig(out_path, dpi=150, facecolor="#111111", bbox_inches="tight")
        plt.close()
        logger.info("SHAP beeswarm written to %s", out_path)

    except Exception as exc:
        logger.warning("SHAP generation failed: %s", exc)
        logger.debug("Full traceback:", exc_info=True)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import joblib
    from utils.config_loader import load_config
    from utils.reproducibility import set_all_seeds

    cfg = load_config(profile_path="profiles/exp1_menos_friccion.yaml", force_reload=True)
    set_all_seeds(cfg["general"]["random_seed"])

    # --- Load serialized matematico model ---
    mat_path = ROOT / "models" / "matematico_v1.joblib"
    if not mat_path.exists():
        logger.error(
            "Model not found at %s — run 'python scripts/setup_models.py' first.", mat_path
        )
        return 1

    matematico_model = joblib.load(mat_path)
    logger.info("Loaded matematico model from %s", mat_path)

    # --- Ensure output directory exists ---
    static_dir = ROOT / "dashboard" / "api" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    # --- Download price data and compute features ---
    logger.info("Downloading price data (this may take a moment)...")
    try:
        from data.downloader import download_all
        from data.features import compute_all_features

        prices = download_all(cfg, force_download=False)
        features = {t: compute_all_features(df, cfg) for t, df in prices.items()}
        logger.info("Features computed for %d tickers", len(features))
    except Exception as exc:
        logger.warning("Could not download features: %s — SHAP will be skipped.", exc)
        features = {}

    # --- Generate SHAP beeswarm ---
    if features:
        generate_shap(static_dir, matematico_model, features, cfg)
    else:
        logger.warning("No features available — skipping SHAP beeswarm.")

    # --- Generate reliability diagram ---
    generate_reliability_diagram(static_dir)

    logger.info("Done. Static files are in %s", static_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
