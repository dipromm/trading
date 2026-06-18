"""
run.py — Punto de entrada del sistema MAS para backtest walk-forward.

Ejecuta el pipeline completo:
    datos -> features -> Matemático -> Juez v1 -> Gestor de Riesgos -> BacktestEngine

Uso:
    python run.py                      # Run con datos cacheados
    python run.py --force-download     # Re-descarga todos los datos
    python run.py --no-baselines       # Saltar comparativa de baselines
    python run.py --debug              # Logging verbose

Salida en consola:
    - Tabla con Sharpe, MaxDD, Return por ventana walk-forward
    - Comparativa contra Buy & Hold y SMA Crossover 20/50

Archivos generados:
    - logs/trades/trades_iter*.jsonl   — log auditable de operaciones
    - experiments/<timestamp>/         — config + métricas del experimento
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("yfinance", "urllib3", "requests", "peewee", "xgboost", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Helpers de presentación ───────────────────────────────────────────────────

def _print_window_table(window_results: list[dict], overall: dict) -> None:
    W = 74
    print("\n" + "=" * W)
    print(f"{'WALK-FORWARD — RESULTADOS OUT-OF-SAMPLE':^{W}}")
    print("=" * W)
    print(
        f"  {'Ventana':<8} {'Período validación':<22}"
        f" {'Sharpe':>7} {'MaxDD%':>7} {'Ret%':>7} {'Win%':>7} {'Tickers':>8}"
    )
    print("-" * W)
    for w in window_results:
        periodo = f"{w['val_start'][:7]} -> {w['val_end'][:7]}"
        print(
            f"  Iter {w['iteration']:<3} {periodo:<22}"
            f" {w['sharpe_ratio']:>7.3f}"
            f" {w['max_drawdown_pct']:>7.1f}"
            f" {w['total_return_pct']:>7.1f}"
            f" {w['win_rate_pct']:>7.1f}"
            f" {w.get('n_tickers', '?'):>8}"
        )
    print("-" * W)
    print(
        f"  {'TOTAL OUT-OF-SAMPLE':<31}"
        f" {overall['sharpe_ratio']:>7.3f}"
        f" {overall['max_drawdown_pct']:>7.1f}"
        f" {overall['total_return_pct']:>7.1f}"
        f" {overall['win_rate_pct']:>7.1f}"
    )
    print("=" * W)


def _print_comparison_table(
    mas_metrics: dict,
    bh_metrics: dict,
    sma_metrics: dict,
) -> None:
    W = 58
    print("\n" + "=" * W)
    print(f"{'COMPARATIVA CON BASELINES (período out-of-sample)':^{W}}")
    print("=" * W)
    print(f"  {'Estrategia':<24} {'Sharpe':>7} {'MaxDD%':>7} {'Ret%':>7}")
    print("-" * W)
    rows = [
        ("MAS — Matemático v1", mas_metrics),
        ("Buy & Hold", bh_metrics),
        ("SMA Crossover 20/50", sma_metrics),
    ]
    for label, m in rows:
        print(
            f"  {label:<24}"
            f" {m['sharpe_ratio']:>7.3f}"
            f" {m['max_drawdown_pct']:>7.1f}"
            f" {m['total_return_pct']:>7.1f}"
        )
    print("=" * W)

    mas_sharpe = mas_metrics["sharpe_ratio"]
    bh_sharpe = bh_metrics["sharpe_ratio"]
    sma_sharpe = sma_metrics["sharpe_ratio"]
    beats_bh = mas_sharpe > bh_sharpe
    beats_sma = mas_sharpe > sma_sharpe

    print()
    print(f"  {'[OK]' if beats_sma else '[!!]'} MAS {'supera' if beats_sma else 'NO supera'} SMA Crossover  "
          f"(delta Sharpe: {mas_sharpe - sma_sharpe:+.3f})")
    print(f"  {'[OK]' if beats_bh else '[!!]'} MAS {'supera' if beats_bh else 'NO supera'} Buy & Hold      "
          f"(delta Sharpe: {mas_sharpe - bh_sharpe:+.3f})")
    print()
    if beats_bh and beats_sma:
        print("  -> Criterio de éxito Fase 3 CUMPLIDO.")
    else:
        print("  -> Criterio de éxito Fase 3 NO cumplido. Revisar features / hiperparámetros.")
    print()


# ── Persistencia de experimento ───────────────────────────────────────────────

def _save_experiment(
    results: dict,
    bh_metrics: dict | None,
    sma_metrics: dict | None,
    config: dict,
) -> Path:
    """Guarda métricas + config en experiments/ para trazabilidad."""
    import shutil
    import yaml

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config["general"]["experiments_dir"]) / f"fase3_matematico_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": timestamp,
        "config_hash": str(Path("config.yaml").stat().st_mtime),
        "mas": {
            "metrics": results["metrics"],
            "window_results": results["window_results"],
        },
    }
    if bh_metrics:
        payload["buy_and_hold"] = bh_metrics
    if sma_metrics:
        payload["sma_crossover"] = sma_metrics

    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    shutil.copy("config.yaml", exp_dir / "config.yaml")

    print(f"  Experimento guardado en: {exp_dir}/")
    return exp_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sistema MAS — Backtest Walk-Forward (Fase 3: Matemático)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-descarga todos los datos ignorando la caché local",
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Saltar el cálculo de baselines (más rápido, sin comparativa)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Logging verbose (nivel DEBUG)",
    )
    args = parser.parse_args()

    _setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # ── 1. Config y reproducibilidad ──────────────────────────────────────
    from utils.config_loader import load_config
    from utils.reproducibility import set_all_seeds

    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    set_all_seeds(seed)

    logger.info(
        "Iniciando run | seed=%d | universo=%s | capital=%.0f€",
        seed,
        cfg["universe"]["tickers_file"],
        cfg["backtester"]["initial_capital"],
    )

    # ── 2. Datos OHLCV ────────────────────────────────────────────────────
    from data.downloader import download_all

    logger.info("Paso 1/4: Cargando datos OHLCV (force=%s)...", args.force_download)
    prices = download_all(cfg, force_download=args.force_download)

    if not prices:
        logger.error("No se obtuvieron datos. Verifica la conexión y el universo CSV.")
        return 1

    logger.info("  %d tickers con datos OK", len(prices))

    # ── 3. Features técnicas ──────────────────────────────────────────────
    from data.features import compute_all_features

    logger.info("Paso 2/4: Calculando features técnicas...")
    features: dict = {}
    for ticker, df in prices.items():
        features[ticker] = compute_all_features(df)
    logger.info("  Features calculadas para %d tickers", len(features))

    # ── 4. Walk-forward ───────────────────────────────────────────────────
    from agents.gestor_riesgos import GestorRiesgos
    from agents.matematico import Matematico
    from backtester.walk_forward import WalkForwardValidator
    from judge.judge_v1 import JuezV1

    matematico = Matematico(cfg)
    gestor = GestorRiesgos.from_config(cfg)
    juez = JuezV1(cfg)
    validator = WalkForwardValidator(cfg)

    n_windows = len(validator.windows)
    logger.info(
        "Paso 3/4: Walk-forward (%d ventanas | train=%d años | val=%d año)...",
        n_windows,
        cfg["backtester"]["walk_forward_train_years"],
        cfg["backtester"]["walk_forward_val_years"],
    )

    results = validator.run(
        agents={"matematico": matematico},
        judge=juez,
        gestor=gestor,
        prices=prices,
        features=features,
        # ticker_classes: se carga automáticamente del CSV del universo
    )

    _print_window_table(results["window_results"], results["metrics"])

    # ── 5. Baselines ──────────────────────────────────────────────────────
    bh_metrics: dict | None = None
    sma_metrics: dict | None = None

    if not args.no_baselines:
        from backtester.metrics import summary
        from baselines import buy_and_hold, sma_crossover

        # Período out-of-sample: mismas fechas que el walk-forward
        oos_start = str(results["daily_returns"].index[0].date())
        oos_end = str(results["daily_returns"].index[-1].date())

        logger.info(
            "Paso 4/4: Calculando baselines [%s -> %s]...", oos_start, oos_end,
        )

        bh_equity, bh_rets = buy_and_hold.run(
            prices, start_date=oos_start, end_date=oos_end, config=cfg,
        )
        bh_metrics = summary(bh_equity, bh_rets)

        sma_equity, sma_rets = sma_crossover.run(
            prices, start_date=oos_start, end_date=oos_end, config=cfg,
        )
        sma_metrics = summary(sma_equity, sma_rets)

        _print_comparison_table(results["metrics"], bh_metrics, sma_metrics)
    else:
        logger.info("Paso 4/4: Baselines omitidos (--no-baselines)")

    # ── 6. Guardar experimento ────────────────────────────────────────────
    _save_experiment(results, bh_metrics, sma_metrics, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
