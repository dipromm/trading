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

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config
        self.windows = generate_windows(config)

    def run(
        self,
        agents: dict,
        judge,
        gestor,
        prices: dict,
        features: dict,
        ticker_classes: dict | None = None,
    ) -> dict:
        """
        Ejecuta el proceso walk-forward completo.

        Args:
            agents: Dict de agentes activos. Debe contener al menos
                    ``{"matematico": Matematico()}``.
                    Opcionalmente: ``{"matematico": ..., "analista": ...}``
                    para Fase 4+ (multi-agente).
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

        Returns:
            dict con:
              - ``equity_curve``: Series encadenada de todo el periodo out-of-sample
              - ``daily_returns``: Series de retornos diarios encadenados
              - ``metrics``: dict con Sharpe, MaxDD, Calmar, WinRate del periodo completo
              - ``window_results``: lista de metricas por ventana
        """
        from backtester.engine import BacktestEngine
        from backtester.metrics import summary

        engine = BacktestEngine(self.config)
        matematico = agents["matematico"]
        analista = agents.get("analista")
        cazador = agents.get("cazador")
        has_analista = analista is not None
        has_cazador = cazador is not None

        active_agents = ["matematico"] + (["analista"] if has_analista else []) + (["cazador"] if has_cazador else [])
        if len(active_agents) > 1:
            logger.info("Multi-agente activo: %s", " + ".join(active_agents))

        if ticker_classes is None:
            ticker_classes = _build_ticker_classes(list(features.keys()), self.config)

        all_returns: list[pd.Series] = []
        window_results: list[dict] = []

        prev_judge_inputs: pd.DataFrame | None = None
        prev_judge_targets: pd.Series | None = None

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
            if prev_judge_inputs is not None and prev_judge_targets is not None:
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

            # -- 6. El Juez emite la senal final --
            veto = pd.Series(
                0,
                index=pd.DataFrame(ticker_probs_mat).index,
                dtype=int,
            )

            final_signals = _combine_agent_signals(
                judge=judge,
                ticker_probs_mat=ticker_probs_mat,
                ticker_probs_ana=ticker_probs_ana,
                ticker_probs_caz=ticker_probs_caz,
                veto=veto,
            )

            # -- 7. Calcular fracciones de Kelly por (fecha, ticker) --
            kelly_records: dict = {}
            for fecha in final_signals.index:
                raw: dict[str, float] = {}
                for ticker in final_signals.columns:
                    p_val = final_signals.loc[fecha, ticker]
                    if pd.isna(p_val):
                        continue
                    b = b_per_ticker.get(ticker, 1.0)
                    ac = ticker_classes.get(ticker, "equity")
                    f = gestor.compute_position_size(float(p_val), b, ac)
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
                            v["cazador"] = int(s.at[fecha])
                    votes_by_date[fecha][ticker] = v
            agent_votes_log = pd.DataFrame(votes_by_date, dtype=object).T

            # -- 9. Ejecutar el backtest de esta ventana --
            engine.reset()
            equity, returns = engine.run(
                prices=prices,
                signals=final_signals,
                kelly_fractions=kelly_df,
                veto_signal=veto,
                atr_data=atr_data,
                agent_votes_log=agent_votes_log,
            )

            if not final_signals.empty:
                last_fecha = final_signals.index[-1]
                last_prices = {
                    t: float(prices[t].loc[last_fecha, "Close"])
                    for t in prices
                    if last_fecha in prices[t].index
                }
                engine.close_all_positions(last_fecha, last_prices)

            engine.save_trade_logs(f"trades_iter{window.iteration:02d}.jsonl")

            all_returns.append(returns)

            win_metrics = summary(equity, returns)
            win_metrics.update({
                "iteration": window.iteration,
                "val_start": window.val_start,
                "val_end": window.val_end,
                "n_tickers": len(ticker_probs_mat),
            })
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
                if ticker in ticker_probs_caz:
                    row["cazador"] = ticker_probs_caz[ticker].reindex(mat_probs.index).fillna(0)
                judge_parts.append(row)
                if ticker in val_feats and "target_binary" in val_feats[ticker].columns:
                    target_parts.append(
                        val_feats[ticker]["target_binary"].reindex(mat_probs.index),
                    )
            prev_judge_inputs = pd.concat(judge_parts) if judge_parts else None
            prev_judge_targets = pd.concat(target_parts) if target_parts else None

        if not all_returns:
            raise RuntimeError(
                "Walk-forward completo sin ninguna ventana de validacion valida. "
                "Verifica que los datos cubren el rango configurado en config.yaml."
            )

        # -- Resultados finales --
        full_returns = pd.concat(all_returns)
        initial_capital = self.config["backtester"]["initial_capital"]
        full_equity = (1 + full_returns).cumprod() * initial_capital

        overall = summary(full_equity, full_returns)

        logger.info(
            "Walk-forward completo: Sharpe=%.3f | MaxDD=%.1f%% | Return=%.1f%% | %d ventanas",
            overall["sharpe_ratio"],
            overall["max_drawdown_pct"],
            overall["total_return_pct"],
            len(window_results),
        )

        return {
            "equity_curve": full_equity,
            "daily_returns": full_returns,
            "metrics": overall,
            "window_results": window_results,
        }
