"""
run.py -- Punto de entrada del sistema MAS para backtest walk-forward.

Ejecuta el pipeline completo:
    datos -> features -> [noticias -> sentimiento] -> agentes -> Juez -> BacktestEngine

Uso:
    python run.py                      # Run con Matematico solo (Fase 3)
    python run.py --analista           # Run con Matematico + Analista (Fase 4)
    python run.py --force-download     # Re-descarga todos los datos
    python run.py --no-baselines       # Saltar comparativa de baselines
    python run.py --debug              # Logging verbose

Salida en consola:
    - Tabla con Sharpe, MaxDD, Return por ventana walk-forward
    - Comparativa contra Buy & Hold y SMA Crossover 20/50

Archivos generados:
    - logs/trades/trades_iter*.jsonl   -- log auditable de operaciones
    - experiments/<timestamp>/         -- config + metricas del experimento
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# -- Logging -------------------------------------------------------------------

def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in (
        "yfinance", "urllib3", "requests", "peewee",
        "xgboost", "numexpr", "transformers", "torch",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# -- Helpers de presentacion ---------------------------------------------------

def _print_window_table(window_results: list[dict], overall: dict) -> None:
    W = 74
    print("\n" + "=" * W)
    print(f"{'WALK-FORWARD -- RESULTADOS OUT-OF-SAMPLE':^{W}}")
    print("=" * W)
    print(
        f"  {'Ventana':<8} {'Periodo validacion':<22}"
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
    mas_label: str = "MAS",
) -> None:
    W = 58
    print("\n" + "=" * W)
    print(f"{'COMPARATIVA CON BASELINES (periodo out-of-sample)':^{W}}")
    print("=" * W)
    print(f"  {'Estrategia':<24} {'Sharpe':>7} {'MaxDD%':>7} {'Ret%':>7}")
    print("-" * W)
    rows = [
        (mas_label, mas_metrics),
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


def _print_phase4_comparison(
    mat_only_metrics: dict | None,
    mat_ana_metrics: dict,
) -> None:
    """Print Phase 4 criterion: does adding the Analista improve over Matematico alone?"""
    if mat_only_metrics is None:
        return

    W = 58
    print("\n" + "=" * W)
    print(f"{'FASE 4: Analista anade valor?':^{W}}")
    print("=" * W)
    print(f"  {'Sistema':<28} {'Sharpe':>7} {'MaxDD%':>7} {'Ret%':>7}")
    print("-" * W)
    rows = [
        ("Matematico solo", mat_only_metrics),
        ("Matematico + Analista", mat_ana_metrics),
    ]
    for label, m in rows:
        print(
            f"  {label:<28}"
            f" {m['sharpe_ratio']:>7.3f}"
            f" {m['max_drawdown_pct']:>7.1f}"
            f" {m['total_return_pct']:>7.1f}"
        )
    print("-" * W)

    delta = mat_ana_metrics["sharpe_ratio"] - mat_only_metrics["sharpe_ratio"]
    improves = delta > 0
    print(
        f"  {'[OK]' if improves else '[!!]'} Delta Sharpe: {delta:+.3f} "
        f"({'MEJORA' if improves else 'NO MEJORA'})"
    )
    print("=" * W)
    print()


# -- Persistencia de experimento -----------------------------------------------

def _save_experiment(
    results: dict,
    bh_metrics: dict | None,
    sma_metrics: dict | None,
    config: dict,
    experiment_name: str = "fase3_matematico",
    extra: dict | None = None,
) -> Path:
    """Guarda metricas + config en experiments/ para trazabilidad."""
    import shutil
    import yaml

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config["general"]["experiments_dir"]) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": timestamp,
        "experiment": experiment_name,
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
    if extra:
        payload.update(extra)

    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    shutil.copy("config.yaml", exp_dir / "config.yaml")

    print(f"  Experimento guardado en: {exp_dir}/")
    return exp_dir


# -- Main ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sistema MAS -- Backtest Walk-Forward",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analista", action="store_true",
        help="Incluir El Analista (FinBERT + noticias). Requiere ALPACA_API_KEY o cache de noticias.",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-descarga todos los datos ignorando la cache local",
    )
    parser.add_argument(
        "--force-sentiment", action="store_true",
        help="Re-computa sentimiento FinBERT ignorando cache",
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Saltar el calculo de baselines (mas rapido, sin comparativa)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Logging verbose (nivel DEBUG)",
    )
    args = parser.parse_args()

    _setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # -- 1. Config y reproducibilidad --
    from utils.config_loader import load_config
    from utils.reproducibility import set_all_seeds

    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    set_all_seeds(seed)

    use_analista = args.analista

    logger.info(
        "Iniciando run | seed=%d | universo=%s | capital=%.0f EUR | analista=%s",
        seed,
        cfg["universe"]["tickers_file"],
        cfg["backtester"]["initial_capital"],
        "SI" if use_analista else "NO",
    )

    # -- 2. Datos OHLCV --
    from data.downloader import download_all, load_tickers

    tickers = load_tickers(cfg)
    logger.info("Paso 1/5: Cargando datos OHLCV (force=%s)...", args.force_download)
    prices = download_all(cfg, force_download=args.force_download)

    if not prices:
        logger.error("No se obtuvieron datos. Verifica la conexion y el universo CSV.")
        return 1

    logger.info("  %d tickers con datos OK", len(prices))

    # -- 3. Features tecnicas --
    from data.features import compute_all_features

    logger.info("Paso 2/5: Calculando features tecnicas...")
    features: dict = {}
    for ticker, df in prices.items():
        features[ticker] = compute_all_features(df)
    logger.info("  Features calculadas para %d tickers", len(features))

    # -- 3b. Noticias + Sentimiento (si --analista) --
    analista_agent = None
    if use_analista:
        logger.info("Paso 3/5: Cargando noticias y precomputando sentimiento...")

        from agents.analista import Analista
        from data.news import download_all_news, get_trading_dates, load_cached_news

        news_data = download_all_news(
            tickers=list(prices.keys()),
            config=cfg,
            force_download=args.force_download,
        )

        if not news_data:
            news_data = load_cached_news(cfg)

        if news_data:
            trading_dates = get_trading_dates(
                cfg["data"]["start_date"],
                cfg["data"]["end_date"],
            )

            analista_agent = Analista(cfg)
            sentiment = analista_agent.precompute_sentiment(
                news_data=news_data,
                trading_dates=trading_dates,
                force=args.force_sentiment,
            )

            features = Analista.merge_sentiment_into_features(features, sentiment)

            coverage = analista_agent.sentiment_coverage(features)
            avg_coverage = coverage["coverage_pct"].mean()
            logger.info(
                "  Cobertura media de sentimiento: %.1f%% (%d tickers con datos)",
                avg_coverage,
                (coverage["coverage_pct"] > 0).sum(),
            )
        else:
            logger.warning(
                "  Sin datos de noticias disponibles. "
                "Ejecutando sin Analista. Configura ALPACA_API_KEY para descarga."
            )
            use_analista = False
    else:
        logger.info("Paso 3/5: Noticias omitidas (sin --analista)")

    # -- 4. Walk-forward --
    from agents.gestor_riesgos import GestorRiesgos
    from agents.matematico import Matematico
    from backtester.walk_forward import WalkForwardValidator
    from judge.judge_v1 import JuezV1

    matematico = Matematico(cfg)
    gestor = GestorRiesgos.from_config(cfg)
    juez = JuezV1(cfg)
    validator = WalkForwardValidator(cfg)

    agents_dict: dict = {"matematico": matematico}
    if use_analista and analista_agent is not None:
        agents_dict["analista"] = analista_agent

    n_agents = len(agents_dict)
    n_windows = len(validator.windows)
    logger.info(
        "Paso 4/5: Walk-forward (%d ventanas | %d agente%s | train=%d anos | val=%d ano)...",
        n_windows,
        n_agents,
        "s" if n_agents > 1 else "",
        cfg["backtester"]["walk_forward_train_years"],
        cfg["backtester"]["walk_forward_val_years"],
    )

    results = validator.run(
        agents=agents_dict,
        judge=juez,
        gestor=gestor,
        prices=prices,
        features=features,
    )

    experiment_name = "fase4_mat_analista" if use_analista else "fase3_matematico"
    mas_label = "MAS (Mat+Analista)" if use_analista else "MAS (Matematico)"

    _print_window_table(results["window_results"], results["metrics"])

    # -- 5. Baselines --
    bh_metrics: dict | None = None
    sma_metrics: dict | None = None

    if not args.no_baselines:
        from backtester.metrics import summary
        from baselines import buy_and_hold, sma_crossover

        oos_start = str(results["daily_returns"].index[0].date())
        oos_end = str(results["daily_returns"].index[-1].date())

        logger.info(
            "Paso 5/5: Calculando baselines [%s -> %s]...", oos_start, oos_end,
        )

        bh_equity, bh_rets = buy_and_hold.run(
            prices, start_date=oos_start, end_date=oos_end, config=cfg,
        )
        bh_metrics = summary(bh_equity, bh_rets)

        sma_equity, sma_rets = sma_crossover.run(
            prices, start_date=oos_start, end_date=oos_end, config=cfg,
        )
        sma_metrics = summary(sma_equity, sma_rets)

        _print_comparison_table(results["metrics"], bh_metrics, sma_metrics, mas_label)
    else:
        logger.info("Paso 5/5: Baselines omitidos (--no-baselines)")

    # -- 5b. Comparativa Fase 4: Analista anade valor? --
    extra_data: dict | None = None
    if use_analista:
        _load_phase3_and_compare(results, cfg)

    # -- 6. Guardar experimento --
    _save_experiment(
        results, bh_metrics, sma_metrics, cfg,
        experiment_name=experiment_name,
    )

    return 0


def _load_phase3_and_compare(results_with_analista: dict, cfg: dict) -> None:
    """
    Try to load the latest Phase 3 experiment results for comparison.
    Prints the Phase 4 comparison table if found.
    """
    exp_dir = Path(cfg["general"]["experiments_dir"])
    if not exp_dir.exists():
        return

    fase3_dirs = sorted(exp_dir.glob("fase3_matematico_*"), reverse=True)
    if not fase3_dirs:
        print("\n  (Sin experimento Fase 3 previo para comparar. "
              "Ejecuta 'python run.py' sin --analista primero.)")
        return

    latest = fase3_dirs[0]
    results_file = latest / "results.json"
    if not results_file.exists():
        return

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            prev = json.load(f)
        mat_only_metrics = prev["mas"]["metrics"]
        _print_phase4_comparison(mat_only_metrics, results_with_analista["metrics"])
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
