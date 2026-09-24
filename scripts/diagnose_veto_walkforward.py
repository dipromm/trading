"""
Diagnóstico walk-forward de tasas de veto del Conspiranoico (Exp8).

Replica las ventanas reales del backtester (train/val por iteración) y mide
% de días con veto / Kelly reducido por ventana. Permite barrer percentiles
del umbral para encontrar pN donde:
    - 2021 / 2023 / 2024: actividad < ~10%
    - 2022: actividad alta pero no extrema (~15–55%, no 66%)

No ejecuta backtest ni optimiza Sharpe de 2022 (anti-overfitting).

Uso:
    python scripts/diagnose_veto_walkforward.py
    python scripts/diagnose_veto_walkforward.py --profile profiles/experiments/exp1_exp3_combined.yaml
    python scripts/diagnose_veto_walkforward.py --sweep 3,5,7,10,12
    python scripts/diagnose_veto_walkforward.py --sweep 3,5,7,10 --output reports/veto_wf.csv
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

for noisy in ("yfinance", "urllib3", "requests"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_section(title: str, width: int = 72) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _activity_pct(stats: dict, soft: bool) -> float:
    if soft:
        return float(stats.get("pct_reduced", 0.0))
    return float(stats.get("pct_veto", 0.0))


def _score_sweep(rows: list[dict]) -> float:
    """
    Penaliza percentiles que violan criterios Exp8.
    Menor score = mejor.
    """
    by_iter = {r["iteration"]: r["activity_pct"] for r in rows}
    penalty = 0.0

    for it in (1, 3, 4):
        pct = by_iter.get(it, 0.0)
        if pct > 10.0:
            penalty += (pct - 10.0) * 2.0

    pct_2022 = by_iter.get(2, 0.0)
    if pct_2022 < 15.0:
        penalty += 15.0 - pct_2022
    if pct_2022 > 55.0:
        penalty += (pct_2022 - 55.0) * 2.0

    return penalty


def run_walkforward_veto_diagnosis(
    cfg: dict,
    prices: dict,
    regime_features: pd.DataFrame,
    veto_percentile: int,
) -> list[dict]:
    from mas.agents.conspiranoico import Conspiranoico
    from mas.backtester.walk_forward import generate_windows

    cfg_run = copy.deepcopy(cfg)
    cfg_run["conspiranoico"]["veto_threshold_percentile"] = veto_percentile

    conspiranoico = Conspiranoico(cfg_run)
    soft = cfg_run["conspiranoico"].get("veto_mode", "binary") == "soft"
    windows = generate_windows(cfg_run)
    results: list[dict] = []

    for window in windows:
        train_regime = regime_features.loc[window.train_start:window.train_end]
        val_regime = regime_features.loc[window.val_start:window.val_end]
        conspiranoico.fit(train_regime)

        if soft:
            risk_scale = conspiranoico.predict_risk_scale(val_regime)
            stats = Conspiranoico.summarize_risk_scale(
                risk_scale,
                severe_scale=conspiranoico._soft_scale_severe,
                moderate_scale=conspiranoico._soft_scale_moderate,
            )
        else:
            veto = conspiranoico.predict(val_regime)
            stats = Conspiranoico.summarize_veto(veto)

        val_year = pd.Timestamp(window.val_start).year
        row = {
            "veto_percentile": veto_percentile,
            "iteration": window.iteration,
            "val_year": val_year,
            "val_start": window.val_start,
            "val_end": window.val_end,
            "activity_pct": _activity_pct(stats, soft),
            **stats,
        }
        results.append(row)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnóstico walk-forward de tasas de veto (Exp8)",
    )
    parser.add_argument(
        "--profile", default=None,
        help="YAML de perfil (ej. profiles/experiments/exp1_exp3_combined.yaml)",
    )
    parser.add_argument(
        "--sweep", default=None,
        help="Percentiles a barrer, separados por coma (ej. 3,5,7,10)",
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV opcional con resultados",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    from mas.utils.config_loader import load_config
    cfg = load_config(profile_path=args.profile, force_reload=True)

    logger.info("Cargando datos OHLCV...")
    from mas.data.downloader import download_all
    prices = download_all(cfg, force_download=args.force_download)
    if not prices:
        logger.error("Sin datos OHLCV.")
        return 1

    from mas.data.regime import build_regime_features, download_vix
    vix = download_vix(cfg, force_download=args.force_download)
    regime_features = build_regime_features(prices, vix, cfg)

    if args.sweep:
        percentiles = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]
    else:
        percentiles = [int(cfg["conspiranoico"].get("veto_threshold_percentile", 5))]

    all_rows: list[dict] = []
    sweep_scores: list[tuple[int, float]] = []

    for p in percentiles:
        rows = run_walkforward_veto_diagnosis(cfg, prices, regime_features, p)
        all_rows.extend(rows)
        sweep_scores.append((p, _score_sweep(rows)))

    soft = cfg["conspiranoico"].get("veto_mode", "binary") == "soft"
    metric_label = "% reducido" if soft else "% veto"

    _print_section(f"WALK-FORWARD VETO — barrido p{percentiles}")
    print(f"  Modo: {cfg['conspiranoico'].get('veto_mode', 'binary')}")
    print(f"  Criterio Exp8: años buenos (2021/23/24) < 10% | 2022 entre 15–55%")
    print()
    print(
        f"  {'p':>4} {'Iter':>5} {'Año':>6} {'Val':>13} "
        f"{metric_label:>10} {'Severo%':>8} {'Mod%':>8}"
    )
    print("  " + "-" * 62)

    for row in all_rows:
        print(
            f"  p{row['veto_percentile']:>2} "
            f"{row['iteration']:>5} "
            f"{row['val_year']:>6} "
            f"{row['val_start'][:4]:>13} "
            f"{row['activity_pct']:>9.1f}% "
            f"{row.get('pct_severe', row.get('pct_veto', 0)):>7.1f}% "
            f"{row.get('pct_moderate', 0):>7.1f}%"
        )

    _print_section("RANKING PERCENTILES (menor penalización = mejor)")
    sweep_scores.sort(key=lambda x: x[1])
    for p, score in sweep_scores:
        flag = " <-- recomendado" if p == sweep_scores[0][0] else ""
        print(f"  p{p:>2}: penalización = {score:.1f}{flag}")

    best_p = sweep_scores[0][0]
    print(f"\n  Percentil sugerido para Exp8: p{best_p}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\n  CSV: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
