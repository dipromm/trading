"""
Validacion Walk-Forward.

Implementa la estrategia de validacion temporal que garantiza resultados
estadisticamente creibles y sin data leakage entre entrenamientos.

Estructura de ventanas (con config por defecto: 3 anos train, 1 ano val):

    ITER 1: ENTRENA [2018-2020] -> VALIDA [2021]
    ITER 2: ENTRENA [2018-2021] -> VALIDA [2022]
    ITER 3: ENTRENA [2018-2022] -> VALIDA [2023]
    ITER 4: ENTRENA [2018-2023] -> VALIDA [2024]
    HOLDOUT [2025]: NO TOCAR hasta que el sistema este finalizado

Las predicciones de validacion de todas las iteraciones se concatenan para
obtener ~4 anos de predicciones out-of-sample sobre las que calcular las metricas finales.

NOTA sobre el Juez:
    En la Iteracion 1, el Juez no tiene datos previos de agentes para entrenarse.
    El Juez en la Iteracion 1 actua como pass-through del Matematico.
    En Iteraciones 2+, el Juez se entrena con las predicciones de los agentes
    de la iteracion anterior.

NOTA sobre multi-agente (Fase 4+):
    Cuando el dict de agentes contiene mas de un agente predictor
    (e.g. matematico + analista + cazador), el walk-forward:
    1. Entrena y predice con cada agente por separado
    2. Entrena al Juez como meta-modelo con las senales combinadas
    3. Predice por ticker combinando las senales de todos los agentes
    4. Fallback al Matematico si el Juez retorna NaN (e.g. sin noticias)
    5. Construye agent_votes_log para logs JSONL auditables (XAI)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Fallback hardcodeado si el CSV no está disponible (solo por robustez)
_ETF_CLASSES_FALLBACK: dict[str, str] = {
    "TLT": "bond", "IEF": "bond",
    "GLD": "gold",
    "XLU": "defensive_equity", "XLP": "defensive_equity",
    "EFA": "international_equity",
}


def _build_ticker_classes(
    tickers: list[str],
    config: dict | None = None,
) -> dict[str, str]:
    """
    Asigna clase de activo a cada ticker.

    Fuente primaria: columna ``asset_class`` del CSV del universo
    (``config.universe.tickers_file``). Esta es la fuente única de verdad
    definida en el Plan, donde cada ticker ya tiene su clase asignada.

    Fallback: mapeo hardcodeado de los ETFs del Plan + 'equity' para el resto.
    """
    if config is not None:
        try:
            tickers_file = Path(config["universe"]["tickers_file"])
            df = pd.read_csv(tickers_file)
            mapping = dict(zip(df["ticker"], df["asset_class"]))
            return {t: mapping.get(t, "equity") for t in tickers}
        except Exception as exc:
            logger.warning("No se pudo leer asset_class del CSV: %s. Usando fallback.", exc)
    return {t: _ETF_CLASSES_FALLBACK.get(t, "equity") for t in tickers}


@dataclass
class WalkForwardWindow:
    """Una ventana individual de entrenamiento + validación."""
    iteration: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str


def generate_windows(config: dict | None = None) -> list[WalkForwardWindow]:
    """
    Genera las ventanas walk-forward según la configuración.

    Args:
        config: Si None, carga config.yaml

    Returns:
        Lista de WalkForwardWindow ordenadas cronológicamente.
    """
    if config is None:
        config = load_config()

    start = pd.Timestamp(config["data"]["start_date"])
    holdout_start = pd.Timestamp(config["data"]["holdout_start"])
    train_years = config["backtester"]["walk_forward_train_years"]
    val_years = config["backtester"]["walk_forward_val_years"]

    windows = []
    iteration = 1
    val_start = start + pd.DateOffset(years=train_years)

    while val_start < holdout_start:
        val_end = val_start + pd.DateOffset(years=val_years)
        if val_end > holdout_start:
            val_end = holdout_start

        windows.append(WalkForwardWindow(
            iteration=iteration,
            train_start=start.strftime("%Y-%m-%d"),
            train_end=val_start.strftime("%Y-%m-%d"),
            val_start=val_start.strftime("%Y-%m-%d"),
            val_end=val_end.strftime("%Y-%m-%d"),
        ))

        iteration += 1
        val_start = val_end

    logger.info("Walk-forward: %d ventanas generadas", len(windows))
    for w in windows:
        logger.info(
            "  Iter %d: train [%s -> %s] | val [%s -> %s]",
            w.iteration, w.train_start, w.train_end, w.val_start, w.val_end,
        )

    return windows


def generate_holdout_window(config: dict | None = None) -> WalkForwardWindow | None:
    """
    Ventana final reservada: entrena con todo el histórico pre-holdout y valida
    desde ``holdout_start`` hasta ``end_date`` (p. ej. 2025).

    Returns:
        None si no hay días de validación holdout (holdout_start >= end_date).
    """
    if config is None:
        config = load_config()

    holdout_start = pd.Timestamp(config["data"]["holdout_start"])
    end_date = pd.Timestamp(config["data"]["end_date"])
    if holdout_start >= end_date:
        return None

    start = config["data"]["start_date"]
    iteration = len(generate_windows(config)) + 1

    return WalkForwardWindow(
        iteration=iteration,
        train_start=start,
        train_end=holdout_start.strftime("%Y-%m-%d"),
        val_start=holdout_start.strftime("%Y-%m-%d"),
        val_end=end_date.strftime("%Y-%m-%d"),
    )


def generate_paper_forward_window(
    config: dict | None = None,
    *,
    val_start: str | pd.Timestamp | None = None,
    val_end: str | pd.Timestamp | None = None,
) -> WalkForwardWindow | None:
    """
    Ventana de inferencia forward para paper trading (2026+).

    Entrena con todo el histórico hasta ``end_date`` (fin del holdout) y genera
    señales desde ``paper_start`` hasta ``val_end`` (por defecto: ayer).
    """
    if config is None:
        config = load_config()

    holdout_end = pd.Timestamp(config["data"]["end_date"])
    paper_start_raw = config["data"].get("paper_start")
    if val_start is not None:
        val_start_ts = pd.Timestamp(val_start)
    elif paper_start_raw:
        val_start_ts = pd.Timestamp(paper_start_raw)
    else:
        val_start_ts = holdout_end + pd.Timedelta(days=1)

    if val_end is None:
        val_end_ts = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    else:
        val_end_ts = pd.Timestamp(val_end)

    if val_start_ts > val_end_ts:
        logger.warning(
            "Ventana paper vacía: paper_start=%s > val_end=%s",
            val_start.date(), val_end_ts.date(),
        )
        return None

    base_windows = generate_windows(config)
    n_extra = 1 if generate_holdout_window(config) is not None else 0
    iteration = len(base_windows) + n_extra + 1

    return WalkForwardWindow(
        iteration=iteration,
        train_start=config["data"]["start_date"],
        train_end=holdout_end.strftime("%Y-%m-%d"),
        val_start=val_start_ts.strftime("%Y-%m-%d"),
        val_end=val_end_ts.strftime("%Y-%m-%d"),
    )


def _combine_agent_signals(
    judge,
    ticker_probs_mat: dict[str, pd.Series],
    ticker_probs_ana: dict[str, pd.Series],
    ticker_probs_caz: dict[str, pd.Series],
    veto: pd.Series,
) -> pd.DataFrame:
    """
    Combine agent predictions through the Juez.

    In pass-through mode (Phase 3 or Iter 1): returns Matematico signals directly.
    In meta-model mode (Phase 4+ Iter 2+): runs per-ticker through the meta-model
    and falls back to Matematico for tickers without secondary-agent coverage.

    Args:
        ticker_probs_caz: {ticker: binary Series 0/1} from El Cazador.
                          Tickers absent from this dict get cazador=0 (silence).
    """
    has_secondary = bool(ticker_probs_ana or ticker_probs_caz)

    if judge.is_pass_through or not has_secondary:
        signals_df = pd.DataFrame(ticker_probs_mat)
        return judge.predict(signals_df, veto_signal=veto)

    per_ticker_signals: dict[str, pd.Series] = {}
    for ticker, mat_probs in ticker_probs_mat.items():
        agent_data = pd.DataFrame({"matematico": mat_probs})
        if ticker in ticker_probs_ana:
            agent_data["analista"] = ticker_probs_ana[ticker].reindex(mat_probs.index)
        elif ticker_probs_ana:
            agent_data["analista"] = np.nan
        if ticker in ticker_probs_caz:
            agent_data["cazador"] = ticker_probs_caz[ticker].reindex(mat_probs.index).fillna(0)
        elif ticker_probs_caz:
            agent_data["cazador"] = 0

        result = judge.predict(agent_data, veto_signal=veto)
        per_ticker_signals[ticker] = result["prob_up"]

    final = pd.DataFrame(per_ticker_signals)

    # Fallback: where the Juez returns NaN (e.g. ticker has no news),
    # use the Matematico's direct signal (still apply veto)
    for ticker in final.columns:
        nan_mask = final[ticker].isna()
        if nan_mask.any() and ticker in ticker_probs_mat:
            fallback = ticker_probs_mat[ticker].reindex(final.index)
            veto_dates = veto.index[veto == 1] if veto is not None else pd.Index([])
            common_veto = fallback.index.intersection(veto_dates)
            if not common_veto.empty:
                fallback = fallback.copy()
                fallback.loc[common_veto] = 0.0
            final[ticker] = final[ticker].fillna(fallback)

    return final


def _apply_cazador_risk_scale(
    kelly_df: pd.DataFrame,
    ticker_alerts: dict[str, pd.Series],
    alert_scale: float,
) -> pd.DataFrame:
    """
    Reduce f* por ticker en días con alerta insider (modo risk_modulator).

    Args:
        kelly_df: Fracciones Kelly {fecha × ticker}.
        ticker_alerts: {ticker: Serie 0/1} de alertas del Cazador.
        alert_scale: Multiplicador aplicado cuando alerta == 1 (ej. 0.5).

    Returns:
        Copia de kelly_df con escalado aplicado.
    """
    if not ticker_alerts or alert_scale >= 1.0:
        return kelly_df

    scaled = kelly_df.copy()
    for ticker, alerts in ticker_alerts.items():
        if ticker not in scaled.columns:
            continue
        mask = alerts.reindex(scaled.index, fill_value=0).astype(bool)
        if mask.any():
            scaled.loc[mask, ticker] *= alert_scale

    return scaled


def _apply_defensive_rotation(
    kelly_df: pd.DataFrame,
    risk_scale: pd.Series,
    veto: pd.Series,
    defensive_tickers: list[str],
    total_allocation: float,
    trigger_scale: float,
    ticker_classes: dict[str, str],
    gestor,
    use_binary_veto: bool,
) -> tuple[pd.DataFrame, int]:
    """
    Exp7: en crisis severa, reduce equity y asigna capital a bonos/oro.

    Modo soft: activa en días con ``risk_scale <= trigger_scale`` (tier severo).
    Modo binary: activa en días con ``veto == 1``.

    Returns:
        (kelly_df actualizado, n_días con rotación aplicada)
    """
    if total_allocation <= 0 or not defensive_tickers:
        return kelly_df, 0

    available = [t for t in defensive_tickers if t in kelly_df.columns]
    if not available:
        return kelly_df, 0

    result = kelly_df.copy()
    equity_tickers = [
        c for c in result.columns
        if ticker_classes.get(c, "equity") == "equity"
    ]

    if use_binary_veto:
        crisis_mask = veto.reindex(result.index, fill_value=0).astype(bool)
    else:
        scale_aligned = risk_scale.reindex(result.index, fill_value=1.0)
        crisis_mask = scale_aligned <= trigger_scale + 1e-9

    crisis_dates = result.index[crisis_mask]
    if len(crisis_dates) == 0:
        return result, 0

    per_ticker = total_allocation / len(available)
    raw_template = {t: per_ticker for t in available}
    tc = {t: ticker_classes.get(t, "bond") for t in available}
    normalized_template = gestor.normalize_portfolio_by_class(raw_template, tc)

    n_rotated = 0
    for fecha in crisis_dates:
        for t in equity_tickers:
            result.loc[fecha, t] = 0.0
        for t, f_star in normalized_template.items():
            result.loc[fecha, t] = f_star
        n_rotated += 1

    return result, n_rotated


class WalkForwardValidator:
    """
    Orquestador del proceso walk-forward completo.

    Conecta el pipeline completo:
        datos -> features -> Matemático -> Juez -> Gestor de Riesgos -> BacktestEngine

    Uso mínimo (Fase 3 — solo el Matemático):
        from data.downloader import download_all
        from data.features import compute_all_features
        from agents.matematico import Matematico
        from judge.judge_v1 import JuezV1
        from agents.gestor_riesgos import GestorRiesgos
        from utils.reproducibility import set_all_seeds

        cfg = load_config()
        set_all_seeds(cfg["general"]["random_seed"])

        prices = download_all(cfg)
        features = {t: compute_all_features(df) for t, df in prices.items()}

        validator = WalkForwardValidator(cfg)
        results = validator.run(
            agents={"matematico": Matematico(cfg)},
            judge=JuezV1(cfg),
            gestor=GestorRiesgos.from_config(cfg),
            prices=prices,
            features=features,
        )
        print(results["metrics"])
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        include_holdout: bool = False,
        include_paper_forward: bool = False,
        paper_forward_only: bool = False,
        paper_val_start: str | pd.Timestamp | None = None,
        paper_val_end: str | pd.Timestamp | None = None,
    ) -> None:
        if config is None:
            config = load_config()
        self.config = config

        if paper_forward_only:
            paper_win = generate_paper_forward_window(
                config, val_start=paper_val_start, val_end=paper_val_end,
            )
            self.windows = [paper_win] if paper_win is not None else []
            if paper_win is not None:
                logger.info(
                    "Paper forward ONLY — Iter %d: val [%s -> %s]",
                    paper_win.iteration, paper_win.val_start, paper_win.val_end,
                )
            return

        self.windows = generate_windows(config)
        if include_holdout:
            holdout = generate_holdout_window(config)
            if holdout is not None:
                self.windows.append(holdout)
                logger.info(
                    "Holdout añadido — Iter %d: train [%s -> %s] | val [%s -> %s]",
                    holdout.iteration,
                    holdout.train_start,
                    holdout.train_end,
                    holdout.val_start,
                    holdout.val_end,
                )
            else:
                logger.warning(
                    "include_holdout=True pero no hay período holdout "
                    "(holdout_start >= end_date)"
                )
        if include_paper_forward:
            paper_win = generate_paper_forward_window(
                config, val_start=paper_val_start, val_end=paper_val_end,
            )
            if paper_win is not None:
                self.windows.append(paper_win)
                logger.info(
                    "Paper forward añadido — Iter %d: train [%s -> %s] | val [%s -> %s]",
                    paper_win.iteration,
                    paper_win.train_start,
                    paper_win.train_end,
                    paper_win.val_start,
                    paper_win.val_end,
                )
            else:
                logger.warning(
                    "include_paper_forward=True pero no hay ventana paper válida "
                    "(paper_start > ayer o sin datos)"
                )

    def run(
        self,
        agents: dict,
        judge,
        gestor,
        prices: dict,
        features: dict,
        ticker_classes: dict | None = None,
        regime_features: "pd.DataFrame | None" = None,
        holdout_cache_dir: "str | Path | None" = None,
        holdout_profile_path: str | None = None,
        paper_cache_dir: "str | Path | None" = None,
        paper_profile_path: str | None = None,
        initial_prev_judge_inputs: "pd.DataFrame | None" = None,
        initial_prev_judge_targets: "pd.Series | None" = None,
        paper_merge_artifacts: bool = False,
        paper_init_engine: bool = True,
    ) -> dict:
        """
        Ejecuta el proceso walk-forward completo.

        Args:
            agents: Dict de agentes activos. Debe contener al menos
                    ``{"matematico": Matematico()}``.
                    Opcionalmente: ``{"matematico": ..., "analista": ...,
                    "cazador": ..., "conspiranoico": ...}``
            judge: Instancia de ``JuezV1``. Comienza en modo pass-through
                   y pasa a meta-modelo a partir de la Iteracion 2 (Fase 4+).
            gestor: Instancia de ``GestorRiesgos``.
            prices: ``{ticker: OHLCV DataFrame}`` de ``download_all()``.
                    Las columnas deben incluir ``"Close"`` (tal como yfinance las entrega).
            features: ``{ticker: features DataFrame}`` de ``compute_all_features()``.
                      Si el Analista esta activo, deben incluir ``sentiment_raw``.
            ticker_classes: ``{ticker: asset_class}`` para caps diferenciados.
                            Si None, se infiere automaticamente: ETFs del Plan
                            se asignan a su clase, el resto es ``"equity"``.
            regime_features: DataFrame de features de régimen de mercado
                             (de ``data.regime.build_regime_features()``).
                             Requerido si ``"conspiranoico"`` está en agents.

        Returns:
            dict con:
              - ``equity_curve``: Series encadenada de todo el periodo out-of-sample
              - ``daily_returns``: Series de retornos diarios encadenados
              - ``metrics``: dict con Sharpe, MaxDD, Calmar, WinRate del periodo completo
              - ``window_results``: lista de metricas por ventana
        """
        matematico = agents["matematico"]
        analista = agents.get("analista")
        cazador = agents.get("cazador")
        conspiranoico = agents.get("conspiranoico")
        has_analista = analista is not None
        has_cazador = cazador is not None
        has_conspiranoico = conspiranoico is not None

        cazador_cfg = self.config.get("cazador", {})
        cazador_mode = cazador_cfg.get("mode", "judge")
        cazador_alert_scale = float(cazador_cfg.get("alert_kelly_scale", 0.5))
        cazador_as_modulator = has_cazador and cazador_mode == "risk_modulator"

        rm_cfg = self.config.get("risk_manager", {})
        boost_cfg = rm_cfg.get("kelly_boost", {})
        kelly_boost_enabled = bool(boost_cfg.get("enabled", False))
        base_kelly_fraction = float(rm_cfg.get("kelly_fraction", 0.5))
        boost_kelly_fraction = float(boost_cfg.get("fraction", 0.6))
        boost_vix_percentile = float(boost_cfg.get("vix_percentile", 70))
        boost_require_normal = bool(boost_cfg.get("require_normal_regime", True))

        judge_cfg = self.config.get("judge_v1", {})
        judge_force_pass_through = judge_cfg.get("mode", "meta_model") == "pass_through"

        con_cfg = self.config.get("conspiranoico", {})
        def_rot_cfg = con_cfg.get("defensive_rotation", {})
        defensive_rotation_enabled = bool(def_rot_cfg.get("enabled", False))
        defensive_tickers = list(def_rot_cfg.get("tickers", ["TLT", "IEF", "GLD"]))
        defensive_total_alloc = float(def_rot_cfg.get("total_allocation", 0.30))
        defensive_trigger_scale = float(
            def_rot_cfg.get("trigger_scale", con_cfg.get("soft_veto_scale_severe", 0.25))
        )

        if kelly_boost_enabled and regime_features is None:
            raise ValueError(
                "kelly_boost requiere 'regime_features' (columna VIX). "
                "Activa --conspiranoico o pasa regime_features al walk-forward."
            )

        if has_conspiranoico and regime_features is None:
            raise ValueError(
                "El Conspiranoico requiere 'regime_features'. "
                "Pasa el DataFrame de data.regime.build_regime_features()."
            )

        from backtester.engine import BacktestEngine
        from backtester.metrics import summary

        engine = BacktestEngine(self.config)

        bt_cfg = self.config.get("backtester", {})
        carry_positions = bool(bt_cfg.get("carry_positions_between_windows", False))
        close_at_window_end = not carry_positions

        active_agents = ["matematico"] + (["analista"] if has_analista else []) + (["cazador"] if has_cazador else []) + (["conspiranoico"] if has_conspiranoico else [])
        if cazador_as_modulator:
            logger.info(
                "Cazador en modo risk_modulator: alertas escalan Kelly ×%.2f (excluido del Juez)",
                cazador_alert_scale,
            )
        if kelly_boost_enabled:
            logger.info(
                "Kelly boost activo: rho=%.2f si VIX < p%.0f train y régimen normal (base=%.2f)",
                boost_kelly_fraction, boost_vix_percentile, base_kelly_fraction,
            )
        if judge_force_pass_through:
            logger.info(
                "Juez en modo pass_through: señal directa del Matemático (Cazador fuera del meta-modelo)",
            )
        if defensive_rotation_enabled:
            logger.info(
                "Rotación defensiva (Exp7): %.0f%% en %s si crisis (scale <= %.2f o veto binario)",
                defensive_total_alloc * 100,
                defensive_tickers,
                defensive_trigger_scale,
            )
        if carry_positions:
            logger.info(
                "Exp10: carry_positions_between_windows=ON — sin cierre forzado ni reset entre ventanas",
            )
        elif close_at_window_end:
            logger.info(
                "Cierre de ventana WF: posiciones liquidadas al final de cada iteración (default)",
            )
        if len(active_agents) > 1:
            logger.info("Multi-agente activo: %s", " + ".join(active_agents))

        if ticker_classes is None:
            ticker_classes = _build_ticker_classes(list(features.keys()), self.config)

        all_returns: list[pd.Series] = []
        all_equity: list[pd.Series] = []
        window_results: list[dict] = []
        window_close_stats: list[dict] = []

        prev_judge_inputs: pd.DataFrame | None = initial_prev_judge_inputs
        prev_judge_targets: pd.Series | None = initial_prev_judge_targets

        for window in self.windows:
            logger.info(
                "=== Iter %d: train [%s -> %s] | val [%s -> %s] ===",
                window.iteration, window.train_start, window.train_end,
                window.val_start, window.val_end,
            )

            # -- 1. Cortar datos por ventana --
            train_feats_per_ticker: dict[str, pd.DataFrame] = {}
            val_feats: dict[str, pd.DataFrame] = {}

            for ticker, df in features.items():
                t_sl = df.loc[window.train_start:window.train_end]
                if not t_sl.empty and str(t_sl.index[-1].date()) >= window.val_start:
                    t_sl = t_sl.iloc[:-1]
                if not t_sl.empty:
                    train_feats_per_ticker[ticker] = t_sl

                v_sl = df.loc[window.val_start:window.val_end]
                if not v_sl.empty:
                    val_feats[ticker] = v_sl

            if not train_feats_per_ticker:
                logger.warning("Iter %d: sin datos de entrenamiento, saltando", window.iteration)
                continue

            # -- 2. Entrenar el Matematico --
            train_all = pd.concat(
                list(train_feats_per_ticker.values()), ignore_index=True,
            )
            matematico.fit(train_all)

            # -- 2b. Entrenar el Analista (si esta activo) --
            analista_active_this_window = False
            if has_analista:
                try:
                    analista.fit(train_all)
                    analista_active_this_window = True
                    logger.info("  Analista entrenado OK")
                except ValueError as exc:
                    logger.warning(
                        "  Analista no pudo entrenarse (datos insuficientes): %s", exc,
                    )

            # -- 2c. El Cazador no se entrena (basado en reglas) --
            if has_cazador:
                cazador.fit(train_all)

            # -- 3. Ratio b por ticker --
            b_per_ticker: dict[str, float] = {}
            for ticker, t_sl in train_feats_per_ticker.items():
                if "return_1d" in t_sl.columns:
                    rets = t_sl["return_1d"].dropna()
                    b_per_ticker[ticker] = (
                        gestor.compute_gain_loss_ratio(rets) if not rets.empty else 1.0
                    )
                else:
                    b_per_ticker[ticker] = 1.0

            # -- 4. Predecir por ticker (Matematico) --
            ticker_probs_mat: dict[str, pd.Series] = {}
            for ticker, v_sl in val_feats.items():
                if v_sl.empty:
                    continue
                ticker_probs_mat[ticker] = matematico.predict(v_sl)

            if not ticker_probs_mat:
                logger.warning("Iter %d: sin predicciones de validacion, saltando", window.iteration)
                continue

            # -- 4b. Predecir por ticker (Analista, si activo) --
            ticker_probs_ana: dict[str, pd.Series] = {}
            if analista_active_this_window:
                for ticker, v_sl in val_feats.items():
                    if v_sl.empty:
                        continue
                    preds = analista.predict(v_sl)
                    non_nan = preds.notna().sum()
                    if non_nan > 0:
                        ticker_probs_ana[ticker] = preds

                logger.info(
                    "  Analista: predicciones para %d/%d tickers",
                    len(ticker_probs_ana), len(ticker_probs_mat),
                )

            # -- 4c. Predecir por ticker (Cazador, si activo) --
            ticker_probs_caz: dict[str, pd.Series] = {}
            if has_cazador:
                alerts_total = 0
                for ticker, v_sl in val_feats.items():
                    if v_sl.empty:
                        continue
                    preds = cazador.predict_ticker(v_sl, ticker)
                    ticker_probs_caz[ticker] = preds
                    alerts_total += int(preds.sum())
                logger.info(
                    "  Cazador: %d alertas en %d tickers",
                    alerts_total, len(ticker_probs_caz),
                )

            # -- 5. Entrenar el Juez con datos de la iteracion anterior --
            if (
                not judge_force_pass_through
                and prev_judge_inputs is not None
                and prev_judge_targets is not None
            ):
                try:
                    judge.fit(prev_judge_inputs, prev_judge_targets)
                    logger.info(
                        "Iter %d: Juez entrenado con predicciones de iter anterior "
                        "(columnas: %s)",
                        window.iteration,
                        list(prev_judge_inputs.columns),
                    )
                except ValueError as exc:
                    logger.debug(
                        "Iter %d: Juez permanece en pass-through -- %s",
                        window.iteration, exc,
                    )

            # -- 5b. Conspiranoico: veto de régimen por ventana --
            val_index = pd.DataFrame(ticker_probs_mat).index
            risk_scale = pd.Series(1.0, index=val_index, dtype=float)
            tier_stats: dict = {}
            if has_conspiranoico and regime_features is not None:
                try:
                    from agents.conspiranoico import Conspiranoico

                    train_regime = regime_features.loc[window.train_start:window.train_end]
                    val_regime = regime_features.loc[window.val_start:window.val_end]
                    conspiranoico.fit(train_regime)

                    if conspiranoico.veto_mode == "soft":
                        risk_scale_raw = conspiranoico.predict_risk_scale(val_regime)
                        risk_scale = risk_scale_raw.reindex(val_index, fill_value=1.0).astype(float)
                        veto = pd.Series(0, index=val_index, dtype=int)
                        tier_stats = Conspiranoico.summarize_risk_scale(
                            risk_scale,
                            severe_scale=conspiranoico._soft_scale_severe,
                            moderate_scale=conspiranoico._soft_scale_moderate,
                        )
                        logger.info(
                            "  Conspiranoico (soft, %s): %d días reducidos (%.1f%%) | "
                            "severo=%.1f%% moderado=%.1f%%",
                            conspiranoico.detector,
                            int(tier_stats["n_severe"] + tier_stats["n_moderate"]),
                            tier_stats["pct_reduced"],
                            tier_stats["pct_severe"],
                            tier_stats["pct_moderate"],
                        )
                    else:
                        veto_raw = conspiranoico.predict(val_regime)
                        veto = veto_raw.reindex(val_index, fill_value=0).astype(int)
                        tier_stats = Conspiranoico.summarize_veto(veto)
                        logger.info(
                            "  Conspiranoico (%s): %d días con veto de %d (%.1f%%)",
                            conspiranoico.detector,
                            tier_stats["n_veto"],
                            tier_stats["n_days"],
                            tier_stats["pct_veto"],
                        )
                except Exception as exc:
                    logger.warning(
                        "  Conspiranoico error en iter %d: %s. Veto = 0, escala = 1.0.",
                        window.iteration, exc,
                    )
                    veto = pd.Series(0, index=val_index, dtype=int)
                    risk_scale = pd.Series(1.0, index=val_index, dtype=float)
            else:
                veto = pd.Series(0, index=val_index, dtype=int)

            # -- 5c. Kelly boost por régimen (Exp5) --
            kelly_fraction_by_date = pd.Series(base_kelly_fraction, index=val_index, dtype=float)
            if kelly_boost_enabled and regime_features is not None:
                from data.regime import compute_kelly_fraction_by_date

                train_regime_kelly = regime_features.loc[window.train_start:window.train_end]
                val_regime_kelly = regime_features.loc[window.val_start:window.val_end]
                kelly_fraction_by_date, _vix_thr = compute_kelly_fraction_by_date(
                    val_index=val_index,
                    train_regime=train_regime_kelly,
                    val_regime=val_regime_kelly,
                    base_fraction=base_kelly_fraction,
                    boost_fraction=boost_kelly_fraction,
                    vix_percentile=boost_vix_percentile,
                    risk_scale=risk_scale,
                    veto=veto,
                    require_normal_regime=boost_require_normal,
                )

            # -- 6. El Juez emite la senal final --
            # En modo risk_modulator o pass_through el Cazador no entra al meta-modelo del Juez.
            caz_for_judge = (
                {}
                if (cazador_as_modulator or judge_force_pass_through)
                else ticker_probs_caz
            )

            final_signals = _combine_agent_signals(
                judge=judge,
                ticker_probs_mat=ticker_probs_mat,
                ticker_probs_ana=ticker_probs_ana,
                ticker_probs_caz=caz_for_judge,
                veto=veto,
            )

            # -- 7. Calcular fracciones de Kelly por (fecha, ticker) --
            kelly_records: dict = {}
            for fecha in final_signals.index:
                rho = float(kelly_fraction_by_date.get(fecha, base_kelly_fraction))
                raw: dict[str, float] = {}
                for ticker in final_signals.columns:
                    p_val = final_signals.loc[fecha, ticker]
                    if pd.isna(p_val):
                        continue
                    b = b_per_ticker.get(ticker, 1.0)
                    ac = ticker_classes.get(ticker, "equity")
                    f = gestor.compute_position_size(
                        float(p_val), b, ac, kelly_fraction=rho,
                    )
                    if f > 0:
                        raw[ticker] = f

                if raw:
                    tc = {t: ticker_classes.get(t, "equity") for t in raw}
                    kelly_records[fecha] = gestor.normalize_portfolio_by_class(raw, tc)
                else:
                    kelly_records[fecha] = {}

            kelly_df = (
                pd.DataFrame.from_dict(kelly_records, orient="index")
                .reindex(columns=final_signals.columns, fill_value=0.0)
                .fillna(0.0)
            )

            # Soft veto: escalar Kelly por anomaly score (sin anular señales del Juez)
            conspiranoico_binary = (
                has_conspiranoico
                and conspiranoico is not None
                and conspiranoico.veto_mode != "soft"
            )
            if has_conspiranoico and conspiranoico is not None and conspiranoico.veto_mode == "soft":
                scale_aligned = risk_scale.reindex(kelly_df.index, fill_value=1.0)
                kelly_df = kelly_df.mul(scale_aligned, axis=0)

            # Exp7: rotación a bonos/oro en crisis severa
            if defensive_rotation_enabled and has_conspiranoico:
                kelly_df, n_rotated = _apply_defensive_rotation(
                    kelly_df=kelly_df,
                    risk_scale=risk_scale,
                    veto=veto,
                    defensive_tickers=defensive_tickers,
                    total_allocation=defensive_total_alloc,
                    trigger_scale=defensive_trigger_scale,
                    ticker_classes=ticker_classes,
                    gestor=gestor,
                    use_binary_veto=conspiranoico_binary,
                )
                if n_rotated > 0:
                    logger.info(
                        "  Rotación defensiva: %d días con asignación a refugio",
                        n_rotated,
                    )

            # Cazador risk_modulator: escalar Kelly por ticker en días con alerta insider
            if cazador_as_modulator and ticker_probs_caz:
                kelly_df = _apply_cazador_risk_scale(
                    kelly_df, ticker_probs_caz, cazador_alert_scale,
                )
                n_scaled = sum(
                    int(alerts.reindex(kelly_df.index, fill_value=0).astype(bool).sum())
                    for alerts in ticker_probs_caz.values()
                )
                logger.info(
                    "  Cazador risk_modulator: %d (fecha,ticker) con Kelly ×%.2f",
                    n_scaled, cazador_alert_scale,
                )

            # -- 8. ATR para stop-loss dinamico --
            atr_cols = {
                t: val_feats[t]["atr"]
                for t in val_feats
                if "atr" in val_feats[t].columns
            }
            atr_data = pd.DataFrame(atr_cols) if atr_cols else None

            # -- 8b. Construir agent_votes_log para logs JSONL auditables --
            votes_by_date: dict = {}
            for fecha in final_signals.index:
                votes_by_date[fecha] = {}
                for ticker in final_signals.columns:
                    v: dict = {}
                    if ticker in ticker_probs_mat:
                        s = ticker_probs_mat[ticker]
                        if fecha in s.index and not pd.isna(s.at[fecha]):
                            v["matematico"] = round(float(s.at[fecha]), 4)
                    if ticker in ticker_probs_ana:
                        s = ticker_probs_ana[ticker]
                        if fecha in s.index and not pd.isna(s.at[fecha]):
                            v["analista"] = round(float(s.at[fecha]), 4)
                    if ticker in ticker_probs_caz:
                        s = ticker_probs_caz[ticker]
                        if fecha in s.index:
                            alert = int(s.at[fecha])
                            v["cazador"] = alert
                            if cazador_as_modulator and alert == 1:
                                v["cazador_scale"] = cazador_alert_scale
                    # El veto/escala es a nivel mercado (mismo para todos los tickers ese día)
                    if fecha in veto.index and int(veto.at[fecha]) == 1:
                        v["conspiranoico"] = 1
                    elif fecha in risk_scale.index and float(risk_scale.at[fecha]) < 1.0:
                        v["conspiranoico"] = round(float(risk_scale.at[fecha]), 2)
                    if kelly_boost_enabled and fecha in kelly_fraction_by_date.index:
                        rho_day = float(kelly_fraction_by_date.at[fecha])
                        if rho_day > base_kelly_fraction:
                            v["kelly_rho"] = rho_day
                    if defensive_rotation_enabled and fecha in kelly_df.index:
                        def_exposure = sum(
                            float(kelly_df.loc[fecha, t])
                            for t in defensive_tickers
                            if t in kelly_df.columns
                        )
                        if def_exposure > 0:
                            v["defensive_rotation"] = round(def_exposure, 3)
                    votes_by_date[fecha][ticker] = v
            agent_votes_log = pd.DataFrame(votes_by_date, dtype=object).T

            # -- 8c. Exportar caches holdout (Tab 1) y paper forward (Tab 4) --
            holdout_start_ts = pd.Timestamp(self.config["data"]["holdout_start"])
            holdout_end_ts = pd.Timestamp(self.config["data"]["end_date"])
            paper_start_raw = self.config["data"].get("paper_start")
            paper_start_ts = (
                pd.Timestamp(paper_start_raw)
                if paper_start_raw
                else holdout_end_ts + pd.Timedelta(days=1)
            )
            val_start_ts = pd.Timestamp(window.val_start)
            val_end_ts = pd.Timestamp(window.val_end)

            is_holdout_window = (
                val_start_ts >= holdout_start_ts and val_end_ts <= holdout_end_ts
            )
            is_paper_window = val_start_ts >= paper_start_ts

            if holdout_cache_dir is not None and is_holdout_window and not is_paper_window:
                from backtester.paper_cache import (
                    config_hash,
                    save_engine_snapshot,
                    save_holdout_artifacts,
                )

                cache_dir = Path(holdout_cache_dir)
                save_holdout_artifacts(
                    cache_dir,
                    final_signals=final_signals,
                    kelly_df=kelly_df,
                    agent_votes_log=agent_votes_log,
                    veto=veto,
                    atr_data=atr_data,
                )
                last_oos = (holdout_start_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                save_engine_snapshot(
                    cache_dir,
                    engine,
                    last_date=last_oos,
                    profile_path=holdout_profile_path or "",
                    profile_hash=config_hash(self.config),
                )
                logger.info(
                    "Holdout cache exportada en %s (%d días)",
                    cache_dir, len(final_signals),
                )

            skip_backtest = False
            if paper_cache_dir is not None and is_paper_window:
                from backtester.paper_cache import (
                    config_hash,
                    save_fresh_paper_engine_snapshot,
                    save_paper_artifacts,
                )

                p_cache = Path(paper_cache_dir)
                save_paper_artifacts(
                    p_cache,
                    final_signals=final_signals,
                    kelly_df=kelly_df,
                    agent_votes_log=agent_votes_log,
                    veto=veto,
                    atr_data=atr_data,
                    merge=paper_merge_artifacts,
                )
                if paper_init_engine:
                    pre_paper = (val_start_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    save_fresh_paper_engine_snapshot(
                        p_cache,
                        self.config,
                        last_date=pre_paper,
                        profile_path=paper_profile_path or "",
                        profile_hash=config_hash(self.config),
                    )
                logger.info(
                    "Paper forward cache exportada en %s (%d días, cartera reset €%.0f)",
                    p_cache,
                    len(final_signals),
                    self.config["backtester"]["initial_capital"],
                )
                skip_backtest = True

            # -- 9. Ejecutar el backtest de esta ventana --
            if skip_backtest:
                win_metrics = {
                    "iteration": window.iteration,
                    "val_start": window.val_start,
                    "val_end": window.val_end,
                    "n_tickers": len(ticker_probs_mat),
                    "paper_forward_only": True,
                }
                window_results.append(win_metrics)
                logger.info(
                    "  Iter %d (paper forward): señales exportadas — backtest diferido al catch-up",
                    window.iteration,
                )
            else:
                if not carry_positions:
                    engine.reset()
                trades_window_start = len(engine.trade_logs)

                equity, returns = engine.run(
                    prices=prices,
                    signals=final_signals,
                    kelly_fractions=kelly_df,
                    veto_signal=veto,
                    atr_data=atr_data,
                    agent_votes_log=agent_votes_log,
                )

                close_stats: dict | None = None
                if close_at_window_end and not final_signals.empty:
                    last_fecha = final_signals.index[-1]
                    last_prices = {
                        t: float(prices[t].loc[last_fecha, "Close"])
                        for t in prices
                        if last_fecha in prices[t].index
                    }
                    close_stats = engine.close_all_positions(last_fecha, last_prices)
                    window_close_stats.append(close_stats)
                    logger.info(
                        "  Cierre ventana: %d posiciones | comisión cierre=%.2f EUR | "
                        "valor previo=%.2f EUR",
                        close_stats["n_positions_closed"],
                        close_stats["commission_closing"],
                        close_stats["portfolio_value_before"],
                    )

                engine.save_trade_logs(
                    f"trades_iter{window.iteration:02d}.jsonl",
                    start_index=trades_window_start,
                )

                all_returns.append(returns)
                if carry_positions:
                    all_equity.append(equity)

                win_metrics = summary(equity, returns)
                win_metrics.update({
                    "iteration": window.iteration,
                    "val_start": window.val_start,
                    "val_end": window.val_end,
                    "n_tickers": len(ticker_probs_mat),
                })
                if tier_stats:
                    win_metrics["conspiranoico_stats"] = tier_stats
                if close_stats:
                    win_metrics["window_close_stats"] = close_stats
                window_results.append(win_metrics)

                logger.info(
                    "  Iter %d -> Sharpe=%.3f | MaxDD=%.1f%% | Return=%.1f%%",
                    window.iteration,
                    win_metrics["sharpe_ratio"],
                    win_metrics["max_drawdown_pct"],
                    win_metrics["total_return_pct"],
                )

            # -- 10. Preparar datos del Juez para la siguiente iteracion --
            judge_parts: list[pd.DataFrame] = []
            target_parts: list[pd.Series] = []
            for ticker, mat_probs in ticker_probs_mat.items():
                row = mat_probs.to_frame(name="matematico")
                if ticker in ticker_probs_ana:
                    row["analista"] = ticker_probs_ana[ticker].reindex(mat_probs.index)
                if ticker in ticker_probs_caz and not cazador_as_modulator and not judge_force_pass_through:
                    row["cazador"] = ticker_probs_caz[ticker].reindex(mat_probs.index).fillna(0)
                judge_parts.append(row)
                if ticker in val_feats and "target_binary" in val_feats[ticker].columns:
                    target_parts.append(
                        val_feats[ticker]["target_binary"].reindex(mat_probs.index),
                    )
            prev_judge_inputs = pd.concat(judge_parts) if judge_parts else None
            prev_judge_targets = pd.concat(target_parts) if target_parts else None

            if paper_cache_dir is not None and is_holdout_window and not is_paper_window:
                from backtester.paper_cache import save_judge_warmstate

                save_judge_warmstate(
                    Path(paper_cache_dir),
                    prev_judge_inputs,
                    prev_judge_targets,
                    profile_hash=config_hash(self.config),
                )

        if not all_returns and not window_results:
            raise RuntimeError(
                "Walk-forward completo sin ninguna ventana de validacion valida. "
                "Verifica que los datos cubren el rango configurado en config.yaml."
            )

        # -- Resultados finales --
        initial_capital = self.config["backtester"]["initial_capital"]
        if carry_positions and all_equity:
            full_equity = pd.concat(all_equity)
            full_returns = full_equity.pct_change().fillna(0)
        else:
            full_returns = pd.concat(all_returns)
            full_equity = (1 + full_returns).cumprod() * initial_capital

        overall = summary(full_equity, full_returns)

        close_summary: dict | None = None
        if window_close_stats:
            close_summary = {
                "n_window_closures": len(window_close_stats),
                "total_positions_closed": sum(
                    s["n_positions_closed"] for s in window_close_stats
                ),
                "total_commission_closing": round(
                    sum(s["commission_closing"] for s in window_close_stats), 2
                ),
            }
            logger.info(
                "Coste cierre ventanas WF: %d cierres | %d posiciones | "
                "comisiones totales=%.2f EUR (%.2f%% del capital inicial)",
                close_summary["n_window_closures"],
                close_summary["total_positions_closed"],
                close_summary["total_commission_closing"],
                close_summary["total_commission_closing"] / initial_capital * 100,
            )

        logger.info(
            "Walk-forward completo: Sharpe=%.3f | MaxDD=%.1f%% | Return=%.1f%% | %d ventanas",
            overall["sharpe_ratio"],
            overall["max_drawdown_pct"],
            overall["total_return_pct"],
            len(window_results),
        )

        result = {
            "equity_curve": full_equity,
            "daily_returns": full_returns,
            "metrics": overall,
            "window_results": window_results,
        }
        if close_summary:
            result["window_close_summary"] = close_summary
        return result
