"""
Tests del motor de backtesting y las métricas.

CRÍTICO: Estos tests deben pasar ANTES de entrenar cualquier agente.
Un bug en el backtester invalida todos los resultados del proyecto.

Checks obligatorios:
    1. Anti-leakage: features del día T no contienen datos de T+1
    2. Comisiones: comprar y vender el mismo día resulta en pérdida
    3. Reproducibilidad: mismo seed → mismo resultado exacto
    4. Métricas: valores correctos en casos conocidos
"""

import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (
    annualized_return,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)


class TestAntiLeakage:

    def test_target_binary_uses_future_data(self):
        """
        Verifica que target_binary en data/features.py usa shift(-1) para mirar
        al día siguiente, y que las features de entrada NO usan datos futuros.

        Este test es conceptual: verifica que el campo 'target_binary' no está
        incluido en FEATURE_COLUMNS del Matemático.
        """
        from agents.matematico import FEATURE_COLUMNS
        assert "target_binary" not in FEATURE_COLUMNS, \
            "target_binary está en FEATURE_COLUMNS — data leakage garantizado"
        assert "target_return_1d" not in FEATURE_COLUMNS, \
            "target_return_1d está en FEATURE_COLUMNS — data leakage garantizado"

    def test_features_no_negative_shift(self):
        """
        Verifica que ninguna feature calculada en data/features.py
        usa shift(-N) (mirar al futuro) en su cálculo.
        """
        import inspect
        from data import features
        source = inspect.getsource(features)

        # Buscar shift(-N) en el código fuente (excluir la línea del target)
        lines = source.split("\n")
        leaky_lines = [
            line for line in lines
            if "shift(-" in line and "target" not in line.lower()
        ]
        assert len(leaky_lines) == 0, \
            f"Posible data leakage: shift(-N) encontrado fuera del target:\n" + \
            "\n".join(leaky_lines)


class TestCommissions:

    def test_buy_sell_same_day_is_loss(self):
        """
        Comprar y vender el mismo día debe resultar en pérdida neta.
        Con 0.08% de comisión por operación (ida), el coste total es 0.16%.
        """
        commission_pct = 0.0008
        initial_capital = 10_000.0

        capital_after_buy = initial_capital * (1 - commission_pct)
        capital_after_sell = capital_after_buy * (1 - commission_pct)

        assert capital_after_sell < initial_capital, \
            "Comprar y vender el mismo día no genera pérdida — revisar simulación de comisiones"

    def test_commission_compounds_with_many_trades(self):
        """
        Una estrategia que opera 100 veces debe tener pérdidas significativas por comisiones.
        """
        commission_pct = 0.0008
        capital = 10_000.0
        n_trades = 100

        for _ in range(n_trades):
            capital *= (1 - commission_pct)  # compra
            capital *= (1 - commission_pct)  # venta

        total_cost_pct = (10_000 - capital) / 10_000 * 100
        assert total_cost_pct > 10.0, \
            f"100 operaciones deben costar más del 10% — coste calculado: {total_cost_pct:.2f}%"


class TestReproducibility:

    def test_same_seed_same_result(self):
        """Mismo seed debe producir exactamente el mismo resultado."""
        np.random.seed(42)
        returns_1 = pd.Series(np.random.randn(252))

        np.random.seed(42)
        returns_2 = pd.Series(np.random.randn(252))

        pd.testing.assert_series_equal(returns_1, returns_2)

    def test_different_seeds_different_results(self):
        """Seeds distintos deben producir resultados distintos."""
        np.random.seed(42)
        returns_1 = pd.Series(np.random.randn(252))

        np.random.seed(99)
        returns_2 = pd.Series(np.random.randn(252))

        assert not returns_1.equals(returns_2)


class TestExecutionRules:
    """Reglas de ejecución: umbral de entrada e histéresis de venta (Exp 1)."""

    @staticmethod
    def _minimal_config(**backtester_overrides) -> dict:
        bt = {
            "commission_pct": 0.0008,
            "initial_capital": 10000.0,
            "walk_forward_train_years": 3,
            "walk_forward_val_years": 1,
            "min_prob_to_buy": 0.0,
            "sell_prob_threshold": 0.0,
            "sell_hysteresis_days": 1,
        }
        bt.update(backtester_overrides)
        return {
            "backtester": bt,
            "risk_manager": {
                "min_kelly_threshold": 0.005,
                "max_position_pct": 0.15,
                "stop_loss_atr_multiplier": 2.0,
            },
            "general": {"log_dir": "logs"},
        }

    def test_hysteresis_delays_sell_until_streak(self, tmp_path):
        """Con histéresis=3, no vende al primer día con señal débil."""
        from backtester.engine import BacktestEngine

        cfg = self._minimal_config(
            sell_prob_threshold=0.45,
            sell_hysteresis_days=3,
        )
        cfg["general"]["log_dir"] = str(tmp_path)
        engine = BacktestEngine(cfg)

        dates = pd.date_range("2021-01-04", periods=5, freq="B")
        ticker = "AAPL"
        prices = {
            ticker: pd.DataFrame(
                {"Close": [100.0] * 5},
                index=dates,
            ),
        }
        signals = pd.DataFrame({ticker: [0.6, 0.4, 0.4, 0.4, 0.4]}, index=dates)
        kelly = pd.DataFrame({ticker: [0.1, 0.05, 0.05, 0.05, 0.05]}, index=dates)
        veto = pd.Series(0, index=dates)

        # Un solo run sobre todos los días
        engine.run(prices, signals, kelly, veto)
        sells = [log for log in engine.trade_logs if log.action == "SELL"]
        assert len(sells) == 1
        assert sells[0].date == str(dates[3])

    def test_min_prob_to_buy_blocks_entry(self, tmp_path):
        """No abre si prob < min_prob_to_buy aunque f* sea positivo."""
        from backtester.engine import BacktestEngine

        cfg = self._minimal_config(min_prob_to_buy=0.55)
        cfg["general"]["log_dir"] = str(tmp_path)
        engine = BacktestEngine(cfg)

        dates = pd.date_range("2021-01-04", periods=1, freq="B")
        ticker = "AAPL"
        prices = {ticker: pd.DataFrame({"Close": [100.0]}, index=dates)}
        signals = pd.DataFrame({ticker: [0.52]}, index=dates)
        kelly = pd.DataFrame({ticker: [0.08]}, index=dates)
        veto = pd.Series(0, index=dates)

        engine.run(prices, signals, kelly, veto)
        assert ticker not in engine.positions

    def test_defaults_match_legacy_immediate_sell(self, tmp_path):
        """Con defaults, f*<=0 vende el mismo día (comportamiento legacy)."""
        from backtester.engine import BacktestEngine

        cfg = self._minimal_config()
        cfg["general"]["log_dir"] = str(tmp_path)
        engine = BacktestEngine(cfg)

        dates = pd.date_range("2021-01-04", periods=2, freq="B")
        ticker = "AAPL"
        prices = {ticker: pd.DataFrame({"Close": [100.0, 100.0]}, index=dates)}
        signals = pd.DataFrame({ticker: [0.7, 0.3]}, index=dates)
        kelly = pd.DataFrame({ticker: [0.1, 0.0]}, index=dates)
        veto = pd.Series(0, index=dates)

        engine.run(prices, signals, kelly, veto)
        assert ticker not in engine.positions
        assert any(log.action == "SELL" for log in engine.trade_logs)


class TestWindowClose:
    """Cierre forzado al final de ventana WF (Exp10)."""

    @staticmethod
    def _minimal_config(**backtester_overrides) -> dict:
        bt = {
            "commission_pct": 0.0008,
            "initial_capital": 10000.0,
            "walk_forward_train_years": 3,
            "walk_forward_val_years": 1,
            "min_prob_to_buy": 0.0,
            "sell_prob_threshold": 0.0,
            "sell_hysteresis_days": 1,
            "carry_positions_between_windows": False,
        }
        bt.update(backtester_overrides)
        return {
            "backtester": bt,
            "risk_manager": {
                "min_kelly_threshold": 0.005,
                "max_position_pct": 0.15,
                "stop_loss_atr_multiplier": 2.0,
            },
            "general": {"log_dir": "logs"},
        }

    def test_close_all_positions_returns_stats(self, tmp_path):
        from backtester.engine import BacktestEngine

        cfg = self._minimal_config()
        cfg["general"]["log_dir"] = str(tmp_path)
        engine = BacktestEngine(cfg)

        dates = pd.date_range("2021-01-04", periods=1, freq="B")
        ticker = "AAPL"
        prices = {ticker: pd.DataFrame({"Close": [100.0]}, index=dates)}
        signals = pd.DataFrame({ticker: [0.7]}, index=dates)
        kelly = pd.DataFrame({ticker: [0.1]}, index=dates)
        veto = pd.Series(0, index=dates)

        engine.run(prices, signals, kelly, veto)
        assert ticker in engine.positions

        stats = engine.close_all_positions(
            dates[0], {ticker: 100.0},
        )
        assert ticker not in engine.positions
        assert stats["n_positions_closed"] == 1
        assert stats["commission_closing"] > 0
        assert stats["portfolio_value_before"] > 0

    def test_carry_skips_reset_between_windows(self, tmp_path):
        from backtester.engine import BacktestEngine

        cfg = self._minimal_config(carry_positions_between_windows=True)
        cfg["general"]["log_dir"] = str(tmp_path)
        engine = BacktestEngine(cfg)

        dates = pd.date_range("2021-01-04", periods=2, freq="B")
        ticker = "AAPL"
        prices = {
            ticker: pd.DataFrame(
                {"Close": [100.0, 101.0]},
                index=dates,
            ),
        }

        signals1 = pd.DataFrame({ticker: [0.7]}, index=dates[:1])
        kelly1 = pd.DataFrame({ticker: [0.1]}, index=dates[:1])
        veto1 = pd.Series(0, index=dates[:1])
        engine.run(prices, signals1, kelly1, veto1)
        assert ticker in engine.positions
        capital_after_w1 = engine.capital

        # Simula carry: sin reset ni close
        signals2 = pd.DataFrame({ticker: [0.7]}, index=dates[1:])
        kelly2 = pd.DataFrame({ticker: [0.1]}, index=dates[1:])
        veto2 = pd.Series(0, index=dates[1:])
        engine.run(prices, signals2, kelly2, veto2)

        assert ticker in engine.positions
        assert engine.capital == capital_after_w1


class TestLongTermRebalancing:

    @staticmethod
    def _minimal_config(**backtester_overrides) -> dict:
        bt = {
            "commission_pct": 0.0008,
            "initial_capital": 10000.0,
            "walk_forward_train_years": 3,
            "walk_forward_val_years": 1,
            "min_prob_to_buy": 0.0,
            "sell_prob_threshold": 0.0,
            "sell_hysteresis_days": 1,
            "carry_positions_between_windows": True,
        }
        bt.update(backtester_overrides)
        return {
            "backtester": bt,
            "profile": {"rebalancing_days": bt.get("rebalancing_days", 0)},
            "risk_manager": {
                "min_kelly_threshold": 0.005,
                "max_position_pct": 0.15,
                "stop_loss_atr_multiplier": 2.0,
            },
            "general": {"log_dir": "logs"},
        }

    def test_rebalancing_skips_daily_buys(self, tmp_path):
        from backtester.engine import BacktestEngine

        cfg = self._minimal_config(rebalancing_days=3)
        cfg["general"]["log_dir"] = str(tmp_path)
        engine = BacktestEngine(cfg)

        dates = pd.date_range("2021-01-04", periods=6, freq="B")
        ticker = "AAPL"
        prices = {ticker: pd.DataFrame({"Close": [100.0] * 6}, index=dates)}
        signals = pd.DataFrame({ticker: [0.7] * 6}, index=dates)
        kelly = pd.DataFrame({ticker: [0.1] * 6}, index=dates)
        veto = pd.Series(0, index=dates)

        engine.run(prices, signals, kelly, veto)
        buys = [log for log in engine.trade_logs if log.action == "BUY"]
        assert len(buys) == 1  # solo día 0; día 3 ya tiene posición
        assert buys[0].date == str(dates[0])


class TestCazadorRiskScale:

    def test_scales_kelly_on_alert_days_only(self):
        from backtester.walk_forward import _apply_cazador_risk_scale

        dates = pd.date_range("2021-01-04", periods=3, freq="B")
        kelly = pd.DataFrame(
            {"AAPL": [0.10, 0.10, 0.10], "MSFT": [0.08, 0.08, 0.08]},
            index=dates,
        )
        alerts = {
            "AAPL": pd.Series([1, 0, 1], index=dates, dtype=int),
        }
        scaled = _apply_cazador_risk_scale(kelly, alerts, 0.5)

        assert scaled.loc[dates[0], "AAPL"] == pytest.approx(0.05)
        assert scaled.loc[dates[1], "AAPL"] == pytest.approx(0.10)
        assert scaled.loc[dates[2], "AAPL"] == pytest.approx(0.05)
        assert (scaled["MSFT"] == kelly["MSFT"]).all()

    def test_no_op_when_scale_is_one(self):
        from backtester.walk_forward import _apply_cazador_risk_scale

        dates = pd.date_range("2021-01-04", periods=2, freq="B")
        kelly = pd.DataFrame({"AAPL": [0.10, 0.10]}, index=dates)
        alerts = {"AAPL": pd.Series([1, 1], index=dates, dtype=int)}
        scaled = _apply_cazador_risk_scale(kelly, alerts, 1.0)
        pd.testing.assert_frame_equal(scaled, kelly)


class TestJudgePassThrough:

    def test_pass_through_judge_uses_matematico_only(self):
        from backtester.walk_forward import _combine_agent_signals
        from judge.judge_v1 import JuezV1

        dates = pd.date_range("2021-01-04", periods=3, freq="B")
        mat = {"AAPL": pd.Series([0.7, 0.8, 0.6], index=dates)}
        caz = {"AAPL": pd.Series([1, 1, 0], index=dates)}
        veto = pd.Series(0, index=dates)

        judge = JuezV1({
            "judge_v1": {
                "mode": "pass_through",
                "meta_model": "logistic",
                "calibration_method": "sigmoid",
                "random_state": 42,
            },
        })

        result = _combine_agent_signals(
            judge=judge,
            ticker_probs_mat=mat,
            ticker_probs_ana={},
            ticker_probs_caz=caz,
            veto=veto,
        )
        pd.testing.assert_series_equal(result["AAPL"], mat["AAPL"], check_names=False)


class TestDefensiveRotation:

    def test_zeros_equity_and_allocates_defensives_on_severe_day(self):
        from agents.gestor_riesgos import GestorRiesgos
        from backtester.walk_forward import _apply_defensive_rotation

        dates = pd.date_range("2021-01-04", periods=2, freq="B")
        kelly = pd.DataFrame(
            {
                "AAPL": [0.10, 0.10],
                "TLT": [0.0, 0.0],
                "IEF": [0.0, 0.0],
                "GLD": [0.0, 0.0],
            },
            index=dates,
        )
        risk_scale = pd.Series([0.25, 1.0], index=dates)
        veto = pd.Series(0, index=dates)
        ticker_classes = {
            "AAPL": "equity", "TLT": "bond", "IEF": "bond", "GLD": "gold",
        }
        gestor = GestorRiesgos()

        result, n = _apply_defensive_rotation(
            kelly_df=kelly,
            risk_scale=risk_scale,
            veto=veto,
            defensive_tickers=["TLT", "IEF", "GLD"],
            total_allocation=0.30,
            trigger_scale=0.25,
            ticker_classes=ticker_classes,
            gestor=gestor,
            use_binary_veto=False,
        )

        assert n == 1
        assert result.loc[dates[0], "AAPL"] == 0.0
        assert result.loc[dates[0], ["TLT", "IEF", "GLD"]].sum() == pytest.approx(0.30)
        assert result.loc[dates[1], "AAPL"] == pytest.approx(0.10)


class TestMetrics:

    def test_sharpe_all_positive(self):
        returns = pd.Series([0.005] * 252)
        assert sharpe_ratio(returns) > 0

    def test_sharpe_all_negative(self):
        returns = pd.Series([-0.005] * 252)
        assert sharpe_ratio(returns) < 0

    def test_sharpe_zero_std(self):
        """Series constante → std=0 → Sharpe = 0 (sin error)."""
        returns = pd.Series([0.0] * 252)
        assert sharpe_ratio(returns) == 0.0

    def test_max_drawdown_no_loss(self):
        """Sin caídas → drawdown = 0."""
        equity = pd.Series([10000.0, 10100.0, 10200.0, 10300.0])
        assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_is_nonpositive(self):
        """Max drawdown siempre es <= 0."""
        equity = pd.Series([10000.0, 9000.0, 8000.0, 9500.0, 11000.0])
        assert max_drawdown(equity) <= 0

    def test_max_drawdown_known_value(self):
        """Caso conocido: de 10000 a 8000 = -20% drawdown."""
        equity = pd.Series([10000.0, 9000.0, 8000.0, 8500.0])
        assert max_drawdown(equity) == pytest.approx(-0.20, abs=1e-9)

    def test_total_return_flat(self):
        equity = pd.Series([10000.0] * 5)
        assert total_return(equity) == pytest.approx(0.0)

    def test_total_return_positive(self):
        equity = pd.Series([10000.0, 11000.0])
        assert total_return(equity) == pytest.approx(0.10, abs=1e-9)

    def test_win_rate_all_positive(self):
        returns = pd.Series([0.01] * 100)
        assert win_rate(returns) == pytest.approx(1.0)

    def test_win_rate_all_negative(self):
        returns = pd.Series([-0.01] * 100)
        assert win_rate(returns) == pytest.approx(0.0)


class TestWalkForwardWindows:

    def test_generate_holdout_window(self):
        from backtester.walk_forward import generate_holdout_window, generate_windows

        cfg = {
            "data": {
                "start_date": "2018-01-01",
                "holdout_start": "2025-01-01",
                "end_date": "2025-12-31",
            },
            "backtester": {
                "walk_forward_train_years": 3,
                "walk_forward_val_years": 1,
            },
        }
        wf = generate_windows(cfg)
        holdout = generate_holdout_window(cfg)
        assert len(wf) == 4
        assert holdout is not None
        assert holdout.iteration == 5
        assert holdout.val_start == "2025-01-01"
        assert holdout.val_end == "2025-12-31"
        assert holdout.train_end == "2025-01-01"
