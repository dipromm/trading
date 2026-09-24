"""
run.py -- Punto de entrada del sistema MAS para backtest walk-forward.

Ejecuta el pipeline completo:
    datos -> features -> [noticias -> sentimiento] -> [insiders] -> agentes -> Juez -> BacktestEngine

Uso:
    python run.py                          # Run con Matematico solo (Fase 3)
    python run.py --analista               # + El Analista (FinBERT, Fase 4)
    python run.py --cazador                # + El Cazador (SEC Form 4, Fase 5)
    python run.py --cazador --conspiranoico --profile profiles/exp1_menos_friccion.yaml
                                           # Configuraci?n can?nica (Fase 6)
    python run.py --force-download         # Re-descarga todos los datos
    python run.py --no-baselines           # Saltar comparativa de baselines
    python run.py --debug                  # Logging verbose

Salida en consola:
    - Tabla con Sharpe, MaxDD, Return por ventana walk-forward
    - Comparativa contra Buy & Hold y SMA Crossover 20/50

Archivos generados:
    - logs/trades/trades_iter*.jsonl   -- log auditable de operaciones (con agent_votes)
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


def _flatten(d: dict, prefix: str = "") -> dict:
    """Aplana un dict anidado a claves 'a.b.c' -> valor."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _log_profile_overrides(logger: logging.Logger, profile_path: str) -> None:
    """Loguea las claves que el perfil YAML sobreescribe respecto al config base."""
    import yaml

    with open(profile_path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}
    overrides.pop("profile", None)
    flat = _flatten(overrides)
    if not flat:
        return
    logger.info("  Overrides del perfil (%d):", len(flat))
    for key, value in sorted(flat.items()):
        logger.info("    %s = %s", key, value)


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
    if results.get("window_close_summary"):
        payload["mas"]["window_close_summary"] = results["window_close_summary"]
    if bh_metrics:
        payload["buy_and_hold"] = bh_metrics
    if sma_metrics:
        payload["sma_crossover"] = sma_metrics
    if extra:
        payload.update(extra)

    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    shutil.copy("config.yaml", exp_dir / "config.yaml")

    # Guardar equity_curve OOS por experimento (para overlay en dashboard)
    equity = results.get("equity_curve")
    if equity is not None and not equity.empty:
        equity.to_csv(exp_dir / "equity_curve.csv", header=["equity"])

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
        "--cazador", action="store_true",
        help="Incluir El Cazador (insiders SEC EDGAR Form 4). Descarga de SEC EDGAR (requiere SEC_USER_AGENT).",
    )
    parser.add_argument(
        "--conspiranoico", action="store_true",
        help="Incluir El Conspiranoico (detección de régimen Isolation Forest). Descarga ^VIX.",
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
    parser.add_argument(
        "--profile", type=str, default=None,
        help="YAML de perfil que sobreescribe config base (ej: profiles/exp1_menos_friccion.yaml)",
    )
    args = parser.parse_args()

    # Credenciales opcionales (ALPACA_*, SEC_USER_AGENT) desde .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    _setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # -- 1. Config y reproducibilidad --
    from mas.utils.config_loader import load_config
    from mas.utils.reproducibility import set_all_seeds

    cfg = load_config(profile_path=args.profile, force_reload=True)
    seed = cfg["general"]["random_seed"]
    set_all_seeds(seed)

    profile_name = cfg.get("profile", {}).get("name")
    profile_suffix = f"_{profile_name}" if profile_name else ""

    use_analista = args.analista
    use_cazador = args.cazador
    use_conspiranoico = args.conspiranoico

    logger.info(
        "Iniciando run | seed=%d | universo=%s | capital=%.0f EUR | "
        "analista=%s | cazador=%s | conspiranoico=%s | perfil=%s",
        seed,
        cfg["universe"]["tickers_file"],
        cfg["backtester"]["initial_capital"],
        "SI" if use_analista else "NO",
        "SI" if use_cazador else "NO",
        "SI" if use_conspiranoico else "NO",
        profile_name or "base",
    )
    if args.profile:
        _log_profile_overrides(logger, args.profile)

    # -- 2. Datos OHLCV --
    from mas.data.downloader import download_all, load_tickers

    tickers = load_tickers(cfg)
    logger.info("Paso 1/5: Cargando datos OHLCV (force=%s)...", args.force_download)
    prices = download_all(cfg, force_download=args.force_download)

    if not prices:
        logger.error("No se obtuvieron datos. Verifica la conexion y el universo CSV.")
        return 1

    logger.info("  %d tickers con datos OK", len(prices))

    # -- 3. Features tecnicas --
    from mas.data.features import compute_all_features, feature_windows_from_config

    logger.info("Paso 2/5: Calculando features tecnicas...")
    feature_windows = feature_windows_from_config(cfg)
    if feature_windows:
        logger.info("  Ventanas de features personalizadas: %s", feature_windows)
    features: dict = {}
    for ticker, df in prices.items():
        features[ticker] = compute_all_features(df, windows=feature_windows)
    logger.info("  Features calculadas para %d tickers", len(features))

    # -- 3b. Noticias + Sentimiento (si --analista) --
    analista_agent = None
    if use_analista:
        logger.info("Paso 3/5: Cargando noticias y precomputando sentimiento...")

        from mas.agents.analista import Analista
        from mas.data.news import download_all_news, get_trading_dates, load_cached_news

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

    # -- 3c. Insiders (si --cazador) --
    cazador_agent = None
    if use_cazador:
        logger.info("Paso 3c/5: Descargando datos de insiders (SEC EDGAR Form 4)...")
        from mas.agents.cazador import Cazador

        cazador_agent = Cazador(cfg)
        try:
            cazador_agent.precompute_insider_signals(
                tickers=list(prices.keys()),
                force_download=args.force_download,
            )
        except Exception as exc:
            logger.warning(
                "  Error descargando insiders: %s. Ejecutando sin Cazador.", exc,
            )
            use_cazador = False
            cazador_agent = None
    else:
        logger.info("Paso 3c/5: Insiders omitidos (sin --cazador)")

    # -- 3d. VIX + features de régimen (si --conspiranoico) --
    regime_features = None
    conspiranoico_agent = None
    if use_conspiranoico:
        logger.info("Paso 3d/5: Descargando VIX y calculando features de régimen...")
        from mas.agents.conspiranoico import Conspiranoico
        from mas.data.regime import build_regime_features, download_vix

        try:
            vix = download_vix(cfg, force_download=args.force_download)
            regime_features = build_regime_features(prices, vix, cfg)
            conspiranoico_agent = Conspiranoico(cfg)
            logger.info(
                "  Features de régimen: %d fechas, %d columnas",
                len(regime_features), len(regime_features.columns),
            )
        except Exception as exc:
            logger.warning(
                "  Error calculando features de régimen: %s. Ejecutando sin Conspiranoico.", exc,
            )
            use_conspiranoico = False
            regime_features = None
            conspiranoico_agent = None
    else:
        logger.info("Paso 3d/5: Régimen omitido (sin --conspiranoico)")

    # -- 4. Walk-forward --
    from mas.agents.gestor_riesgos import GestorRiesgos
    from mas.agents.matematico import Matematico
    from mas.backtester.walk_forward import WalkForwardValidator
    from mas.judge.judge_v1 import JuezV1

    matematico = Matematico(cfg)
    gestor = GestorRiesgos.from_config(cfg)
    juez = JuezV1(cfg)
    validator = WalkForwardValidator(cfg)

    agents_dict: dict = {"matematico": matematico}
    if use_analista and analista_agent is not None:
        agents_dict["analista"] = analista_agent
    if use_cazador and cazador_agent is not None:
        agents_dict["cazador"] = cazador_agent
    if use_conspiranoico and conspiranoico_agent is not None:
        agents_dict["conspiranoico"] = conspiranoico_agent

    n_agents = len(agents_dict)
    agent_names = " + ".join(
        k.capitalize() for k in ["matematico", "analista", "cazador"] if k in agents_dict
    )
    n_windows = len(validator.windows)
    logger.info(
        "Paso 4/5: Walk-forward (%d ventanas | %s | train=%d anos | val=%d ano)...",
        n_windows,
        agent_names,
        cfg["backtester"]["walk_forward_train_years"],
        cfg["backtester"]["walk_forward_val_years"],
    )

    results = validator.run(
        agents=agents_dict,
        judge=juez,
        gestor=gestor,
        prices=prices,
        features=features,
        regime_features=regime_features,
    )

    if use_conspiranoico and use_cazador and use_analista:
        experiment_name = "fase6_mat_analista_cazador_conspiranoico"
        mas_label = "MAS (Mat+Analista+Cazador+Conspiranoico)"
    elif use_conspiranoico and use_cazador:
        experiment_name = f"fase6_mat_cazador_conspiranoico{profile_suffix}"
        mas_label = "MAS (Mat+Cazador+Conspiranoico)" + (f" [{profile_name}]" if profile_name else "")
    elif use_conspiranoico and use_analista:
        experiment_name = "fase6_mat_analista_conspiranoico"
        mas_label = "MAS (Mat+Analista+Conspiranoico)"
    elif use_conspiranoico:
        experiment_name = "fase6_mat_conspiranoico"
        mas_label = "MAS (Mat+Conspiranoico)"
    elif use_analista and use_cazador:
        experiment_name = "fase5_mat_analista_cazador"
        mas_label = "MAS (Mat+Analista+Cazador)"
    elif use_cazador:
        experiment_name = "fase5_mat_cazador"
        mas_label = "MAS (Mat+Cazador)"
    elif use_analista:
        experiment_name = "fase4_mat_analista"
        mas_label = "MAS (Mat+Analista)"
    else:
        experiment_name = "fase3_matematico"
        mas_label = "MAS (Matematico)"

    _print_window_table(results["window_results"], results["metrics"])

    # -- 5. Baselines --
    bh_metrics: dict | None = None
    sma_metrics: dict | None = None

    if not args.no_baselines:
        from mas.backtester.metrics import summary
        from mas.baselines import buy_and_hold, sma_crossover

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

    # -- 5b. Comparativas entre fases --
    if use_conspiranoico:
        _load_fase5_and_compare(results, cfg, use_cazador=use_cazador, use_analista=use_analista)
    elif use_analista and use_cazador:
        _load_phase4_and_compare(results, cfg)
    elif use_analista:
        _load_phase3_and_compare(results, cfg)

    # -- 6. Guardar experimento --
    _save_experiment(
        results, bh_metrics, sma_metrics, cfg,
        experiment_name=experiment_name,
    )

    active_agents = []
    if use_analista:
        active_agents.append("analista")
    if use_cazador:
        active_agents.append("cazador")
    if use_conspiranoico:
        active_agents.append("conspiranoico")

    _save_dashboard_artifacts(
        results, cfg,
        experiment_name=experiment_name,
        mas_label=mas_label,
        profile_path=args.profile,
        agents=active_agents,
    )

    return 0


def _save_dashboard_artifacts(
    results: dict,
    config: dict,
    *,
    experiment_name: str,
    mas_label: str,
    profile_path: str | None = None,
    agents: list[str] | None = None,
) -> None:
    """Exporta curva de capital y métricas para el dashboard Streamlit."""
    dash_dir = Path(config["general"]["log_dir"]) / "dashboard"
    dash_dir.mkdir(parents=True, exist_ok=True)

    equity = results.get("equity_curve")
    if equity is not None and not equity.empty:
        equity.to_csv(dash_dir / "equity_curve.csv", header=["equity"])

    metrics = results.get("metrics", {})
    with open(dash_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    meta = {
        "experiment": experiment_name,
        "mas_label": mas_label,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "n_trading_days": metrics.get("n_trading_days"),
        "profile_path": profile_path,
        "agents": agents or [],
    }
    with open(dash_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger = logging.getLogger(__name__)
    logger.info("Artefactos del dashboard guardados en %s/", dash_dir)


def _load_phase4_and_compare(results_with_cazador: dict, cfg: dict) -> None:
    """
    Compara el experimento Fase 5 contra el último experimento Fase 4.
    Imprime tabla de comparativa si hay datos disponibles.
    """
    exp_dir = Path(cfg["general"]["experiments_dir"])
    if not exp_dir.exists():
        return

    fase4_dirs = sorted(exp_dir.glob("fase4_mat_analista_*"), reverse=True)
    if not fase4_dirs:
        print("\n  (Sin experimento Fase 4 previo para comparar. "
              "Ejecuta 'python run.py --analista' primero.)")
        return

    latest = fase4_dirs[0]
    results_file = latest / "results.json"
    if not results_file.exists():
        return

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            prev = json.load(f)
        fase4_metrics = prev["mas"]["metrics"]
        W = 58
        print("\n" + "=" * W)
        print(f"{'FASE 5: Cazador anade valor sobre Analista?':^{W}}")
        print("=" * W)
        print(f"  {'Sistema':<28} {'Sharpe':>7} {'MaxDD%':>7} {'Ret%':>7}")
        print("-" * W)
        rows = [
            ("Mat+Analista (Fase 4)", fase4_metrics),
            ("Mat+Analista+Cazador (Fase 5)", results_with_cazador["metrics"]),
        ]
        for label, m in rows:
            print(
                f"  {label:<28}"
                f" {m['sharpe_ratio']:>7.3f}"
                f" {m['max_drawdown_pct']:>7.1f}"
                f" {m['total_return_pct']:>7.1f}"
            )
        print("-" * W)
        delta = results_with_cazador["metrics"]["sharpe_ratio"] - fase4_metrics["sharpe_ratio"]
        improves = delta > 0
        print(
            f"  {'[OK]' if improves else '[!!]'} Delta Sharpe: {delta:+.3f} "
            f"({'MEJORA' if improves else 'NO MEJORA'})"
        )
        print("=" * W)
        print()
    except Exception:
        pass


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


def _load_fase5_and_compare(
    results_with_conspiranoico: dict,
    cfg: dict,
    use_cazador: bool = True,
    use_analista: bool = False,
) -> None:
    """
    Compara Fase 6 (con Conspiranoico) contra el último experimento Fase 5
    equivalente (Mat+Cazador o Mat+Analista+Cazador) para medir el impacto del veto.
    """
    exp_dir = Path(cfg["general"]["experiments_dir"])
    if not exp_dir.exists():
        return

    # Buscar el experimento Fase 5 más reciente del mismo conjunto de agentes base
    if use_analista and use_cazador:
        pattern = "fase5_mat_analista_cazador_*"
        label_prev = "Mat+Analista+Cazador (Fase 5)"
    elif use_cazador:
        pattern = "fase5_mat_cazador_*"
        label_prev = "Mat+Cazador (Fase 5)"
    elif use_analista:
        pattern = "fase4_mat_analista_*"
        label_prev = "Mat+Analista (Fase 4)"
    else:
        pattern = "fase3_matematico_*"
        label_prev = "Matematico (Fase 3)"

    prev_dirs = sorted(exp_dir.glob(pattern), reverse=True)
    if not prev_dirs:
        print(f"\n  (Sin experimento previo '{pattern}' para comparar.)")
        return

    results_file = prev_dirs[0] / "results.json"
    if not results_file.exists():
        return

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            prev = json.load(f)
        prev_metrics = prev["mas"]["metrics"]
        curr_metrics = results_with_conspiranoico["metrics"]

        W = 64
        print("\n" + "=" * W)
        print(f"{'FASE 6: Conspiranoico añade valor?':^{W}}")
        print("=" * W)
        print(f"  {'Sistema':<34} {'Sharpe':>7} {'MaxDD%':>7} {'Ret%':>7}")
        print("-" * W)
        for label, m in [(label_prev, prev_metrics), (f"{label_prev.split('(')[1].rstrip(')')}+Conspiranoico", curr_metrics)]:
            print(
                f"  {label:<34}"
                f" {m['sharpe_ratio']:>7.3f}"
                f" {m['max_drawdown_pct']:>7.1f}"
                f" {m['total_return_pct']:>7.1f}"
            )
        print("-" * W)
        delta_sharpe = curr_metrics["sharpe_ratio"] - prev_metrics["sharpe_ratio"]
        delta_dd = curr_metrics["max_drawdown_pct"] - prev_metrics["max_drawdown_pct"]
        improves_sharpe = delta_sharpe >= -0.05
        improves_dd = delta_dd > 0  # menos negativo = mejor
        print(
            f"  {'[OK]' if improves_sharpe else '[!!]'} Delta Sharpe:  {delta_sharpe:+.3f} "
            f"({'OK' if improves_sharpe else 'DEGRADA'})"
        )
        print(
            f"  {'[OK]' if improves_dd else '[!!]'} Delta MaxDD%:  {delta_dd:+.1f} pp "
            f"({'MEJORA' if improves_dd else 'EMPEORA'})"
        )
        print("=" * W)
        print()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
