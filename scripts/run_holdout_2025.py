#!/usr/bin/env python3
"""
Validación holdout 2025 — Exp1 (período reservado, no usado en diseño WF).

Ejecuta walk-forward 2021-2024 + ventana holdout 2025 con carry entre ventanas.
Reporta métricas OOS (pre-holdout) vs holdout 2025 por separado.

Uso:
    python scripts/run_holdout_2025.py
    python scripts/run_holdout_2025.py --profile profiles/exp1_menos_friccion.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _metrics_for_period(equity, returns, *, before: str | None = None, from_date: str | None = None) -> dict:
    from backtester.metrics import summary

    if before is not None:
        cut = pd.Timestamp(before)
        eq = equity.loc[equity.index < cut]
        rets = returns.loc[returns.index < cut]
    elif from_date is not None:
        cut = pd.Timestamp(from_date)
        eq = equity.loc[equity.index >= cut]
        rets = returns.loc[returns.index >= cut]
    else:
        eq, rets = equity, returns

    if eq.empty or rets.empty:
        return {}
    return summary(eq, rets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Holdout 2025 — Exp1 + carry")
    parser.add_argument(
        "--profile",
        default="profiles/exp1_menos_friccion.yaml",
        help="Perfil YAML (default: Exp1 menos fricción)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Forzar re-descarga de datos",
    )
    args = parser.parse_args()

    from utils.config_loader import load_config
    from utils.reproducibility import set_all_seeds

    cfg = load_config(profile_path=args.profile, force_reload=True)
    set_all_seeds(cfg["general"]["random_seed"])

    holdout_start = cfg["data"]["holdout_start"]
    end_date = cfg["data"]["end_date"]
    profile_name = cfg.get("profile", {}).get("name", Path(args.profile).stem)

    logging.basicConfig(
        level=getattr(logging, cfg["general"]["log_level"], logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(
        "Holdout 2025 | perfil=%s | OOS hasta %s | holdout [%s -> %s]",
        profile_name,
        holdout_start,
        holdout_start,
        end_date,
    )

    # -- Datos (mismo pipeline que run.py Fase 6) --
    from data.downloader import download_all
    from data.features import compute_all_features
    from data.regime import build_regime_features, download_vix
    from agents.cazador import Cazador
    from agents.conspiranoico import Conspiranoico
    from agents.gestor_riesgos import GestorRiesgos
    from agents.matematico import Matematico
    from backtester.walk_forward import WalkForwardValidator
    from baselines import buy_and_hold
    from judge.judge_v1 import JuezV1

    logger.info("Cargando precios y features...")
    prices = download_all(cfg, force_download=args.force_download)
    features = {t: compute_all_features(df, cfg) for t, df in prices.items()}
    logger.info("  %d tickers OK", len(prices))

    vix = download_vix(cfg, force_download=args.force_download)
    regime_features = build_regime_features(prices, vix, cfg)

    matematico = Matematico(cfg)
    gestor = GestorRiesgos.from_config(cfg)
    juez = JuezV1(cfg)
    cazador = Cazador(cfg)
    conspiranoico = Conspiranoico(cfg)

    validator = WalkForwardValidator(cfg, include_holdout=True)
    logger.info(
        "Walk-forward + holdout: %d ventanas (última = holdout 2025)",
        len(validator.windows),
    )

    results = validator.run(
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

    equity = results["equity_curve"]
    returns = results["daily_returns"]
    holdout_ts = holdout_start

    oos_metrics = _metrics_for_period(equity, returns, before=holdout_ts)
    holdout_metrics = _metrics_for_period(equity, returns, from_date=holdout_ts)

    bh_holdout_eq, bh_holdout_rets = buy_and_hold.run(
        prices, start_date=holdout_start, end_date=end_date, config=cfg,
    )
    from backtester.metrics import summary

    bh_holdout = summary(bh_holdout_eq, bh_holdout_rets) if not bh_holdout_eq.empty else {}

    holdout_window = next(
        (w for w in results["window_results"] if w.get("val_start", "") >= holdout_start),
        None,
    )
    oos_windows = [w for w in results["window_results"] if w.get("val_start", "") < holdout_start]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile_name,
        "profile_file": args.profile,
        "holdout_start": holdout_start,
        "end_date": end_date,
        "carry_positions": cfg["backtester"].get("carry_positions_between_windows", False),
        "oos_2021_2024": {
            "period": f"{str(equity.index[0].date())} -> {holdout_start}",
            "n_windows": len(oos_windows),
            "metrics": oos_metrics,
            "windows": oos_windows,
        },
        "holdout_2025": {
            "period": f"{holdout_start} -> {end_date}",
            "mas_metrics": holdout_metrics,
            "buy_and_hold": bh_holdout,
            "delta_sharpe_vs_bh": round(
                holdout_metrics.get("sharpe_ratio", 0) - bh_holdout.get("sharpe_ratio", 0),
                3,
            ),
            "window_result": holdout_window,
        },
        "full_run_including_holdout": results["metrics"],
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"holdout_2025_{profile_name}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- Consola --
    W = 62
    print("\n" + "=" * W)
    print(f"{'HOLDOUT 2025 — ' + profile_name:^62}")
    print("=" * W)
    print(f"  OOS (2021-2024, {len(oos_windows)} ventanas WF)")
    if oos_metrics:
        print(
            f"    Sharpe={oos_metrics['sharpe_ratio']:.3f} | "
            f"Return={oos_metrics['total_return_pct']:+.1f}% | "
            f"MaxDD={oos_metrics['max_drawdown_pct']:.1f}%"
        )
    print(f"  Holdout [{holdout_start} -> {end_date}] (carry desde fin 2024)")
    if holdout_metrics:
        print(
            f"    MAS  Sharpe={holdout_metrics['sharpe_ratio']:.3f} | "
            f"Return={holdout_metrics['total_return_pct']:+.1f}% | "
            f"MaxDD={holdout_metrics['max_drawdown_pct']:.1f}%"
        )
    if bh_holdout:
        print(
            f"    B&H  Sharpe={bh_holdout['sharpe_ratio']:.3f} | "
            f"Return={bh_holdout['total_return_pct']:+.1f}% | "
            f"MaxDD={bh_holdout['max_drawdown_pct']:.1f}%"
        )
        delta = report["holdout_2025"]["delta_sharpe_vs_bh"]
        print(f"    Delta Sharpe MAS vs B&H (2025): {delta:+.3f}")
    if holdout_window:
        print(
            f"    Ventana WF iter {holdout_window.get('iteration')}: "
            f"Ret={holdout_window.get('total_return_pct'):+.1f}% | "
            f"veto={holdout_window.get('conspiranoico_stats', {})}"
        )
    print("=" * W)
    print(f"  Informe: {out_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
