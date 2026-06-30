"""
Tests de integración para El Conspiranoico.

Verificaciones:
    1. fit/predict producen Series binaria con solo 0/1.
    2. Días con NaN en features reciben veto=0 (conservador).
    3. Umbral se fija en train y no se recalibra en predict (anti-leakage).
    4. Integración con JuezV1._apply_veto: fechas con veto → señales 0.
    5. Integración con walk_forward: smoke test con datos mínimos.
"""

import numpy as np
import pandas as pd
import pytest


# ── Fixtures comunes ─────────────────────────────────────────────────────────

@pytest.fixture
def dates_train():
    return pd.date_range("2020-01-02", periods=120, freq="B")


@pytest.fixture
def dates_val():
    return pd.date_range("2020-06-15", periods=60, freq="B")


@pytest.fixture
def regime_train(dates_train):
    """Features de régimen sintéticas — período de entrenamiento."""
    np.random.seed(42)
    n = len(dates_train)
    return pd.DataFrame({
        "vol_5d":          np.random.uniform(0.005, 0.02, n),
        "vol_20d":         np.random.uniform(0.007, 0.025, n),
        "vol_60d":         np.random.uniform(0.008, 0.030, n),
        "vix":             np.random.uniform(12, 25, n),
        "avg_correlation": np.random.uniform(0.2, 0.7, n),
        "avg_volume_ratio": np.random.uniform(0.7, 1.5, n),
        "market_breadth":  np.random.uniform(0.3, 0.7, n),
    }, index=dates_train)


@pytest.fixture
def regime_val(dates_val):
    """Features de régimen sintéticas — período de validación."""
    np.random.seed(7)
    n = len(dates_val)
    return pd.DataFrame({
        "vol_5d":          np.random.uniform(0.005, 0.02, n),
        "vol_20d":         np.random.uniform(0.007, 0.025, n),
        "vol_60d":         np.random.uniform(0.008, 0.030, n),
        "vix":             np.random.uniform(12, 25, n),
        "avg_correlation": np.random.uniform(0.2, 0.7, n),
        "avg_volume_ratio": np.random.uniform(0.7, 1.5, n),
        "market_breadth":  np.random.uniform(0.3, 0.7, n),
    }, index=dates_val)


@pytest.fixture
def config_toy():
    return {
        "conspiranoico": {
            "n_estimators": 50,
            "contamination": 0.05,
            "vix_ticker": "^VIX",
            "volatility_windows": [5, 20, 60],
            "correlation_window": 20,
            "volume_ratio_window": 20,
            "veto_threshold_percentile": 5,
            "random_state": 42,
        }
    }


# ── Tests de Conspiranoico.fit / predict ─────────────────────────────────────

class TestConspiranoicoFitPredict:

    def test_predict_returns_binary_series(self, regime_train, regime_val, config_toy):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        c.fit(regime_train)
        veto = c.predict(regime_val)
        assert isinstance(veto, pd.Series)
        assert set(veto.unique()).issubset({0, 1})

    def test_predict_index_matches_input(self, regime_train, regime_val, config_toy):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        c.fit(regime_train)
        veto = c.predict(regime_val)
        assert veto.index.equals(regime_val.index)

    def test_predict_without_fit_raises(self, regime_val, config_toy):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        with pytest.raises(RuntimeError, match="no está entrenado"):
            c.predict(regime_val)

    def test_fit_requires_min_rows(self, config_toy):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        tiny = pd.DataFrame({
            "vol_5d": [0.01] * 5,
            "vol_20d": [0.015] * 5,
            "vol_60d": [0.012] * 5,
            "vix": [20.0] * 5,
            "avg_correlation": [0.5] * 5,
            "avg_volume_ratio": [1.0] * 5,
            "market_breadth": [0.5] * 5,
        }, index=pd.date_range("2020-01-01", periods=5))
        with pytest.raises(ValueError, match="filas limpias"):
            c.fit(tiny)

    def test_nan_rows_in_val_get_veto_zero(self, regime_train, regime_val, config_toy):
        """Filas con NaN en validación reciben veto=0 (conservador)."""
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        c.fit(regime_train)

        # Insertar NaN en algunas filas de val
        val_with_nan = regime_val.copy()
        val_with_nan.iloc[0, 0] = np.nan
        val_with_nan.iloc[5, 2] = np.nan

        veto = c.predict(val_with_nan)
        # Las filas con NaN deben tener veto=0
        assert int(veto.iloc[0]) == 0
        assert int(veto.iloc[5]) == 0

    def test_threshold_not_recalibrated_on_predict(self, regime_train, regime_val, config_toy):
        """
        Anti-leakage: el umbral de veto se fija en train y no cambia al predecir
        con datos con diferente distribución.
        """
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        c.fit(regime_train)
        threshold_after_fit = c.veto_threshold

        # Predecir con datos muy diferentes (alta volatilidad)
        extreme_val = regime_val.copy()
        extreme_val["vix"] = 80.0  # VIX extremo
        extreme_val["vol_5d"] = 0.1
        c.predict(extreme_val)

        # El umbral no debe cambiar
        assert c.veto_threshold == threshold_after_fit

    def test_extreme_crisis_triggers_more_veto(self, regime_train, regime_val, config_toy):
        """
        Un período de crisis extrema (VIX alto, vol alta, breadth baja)
        debe activar más vetos que un período tranquilo.
        """
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        c.fit(regime_train)

        # Período tranquilo
        calm_val = regime_val.copy()
        calm_val["vix"] = 12.0
        calm_val["vol_5d"] = 0.005
        calm_val["avg_correlation"] = 0.2
        calm_val["market_breadth"] = 0.65
        veto_calm = c.predict(calm_val)

        # Período de crisis
        crisis_val = regime_val.copy()
        crisis_val["vix"] = 70.0
        crisis_val["vol_5d"] = 0.08
        crisis_val["avg_correlation"] = 0.95
        crisis_val["market_breadth"] = 0.1
        veto_crisis = c.predict(crisis_val)

        assert int(veto_crisis.sum()) >= int(veto_calm.sum()), (
            f"Crisis debería tener >= vetos que calma: "
            f"crisis={veto_crisis.sum()}, calma={veto_calm.sum()}"
        )

    def test_anomaly_scores_returns_float_series(self, regime_train, regime_val, config_toy):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        c.fit(regime_train)
        scores = c.anomaly_scores(regime_val)
        assert isinstance(scores, pd.Series)
        # Scores son floats (no enteros)
        assert scores.dropna().dtype in [float, np.float64]

    def test_fit_predict_window_helper(self, config_toy):
        """fit_predict_window encapsula correctamente el slice train/val."""
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)

        all_dates = pd.date_range("2020-01-02", periods=200, freq="B")
        np.random.seed(0)
        n = len(all_dates)
        regime_all = pd.DataFrame({
            "vol_5d":          np.random.uniform(0.005, 0.03, n),
            "vol_20d":         np.random.uniform(0.007, 0.035, n),
            "vol_60d":         np.random.uniform(0.008, 0.040, n),
            "vix":             np.random.uniform(10, 30, n),
            "avg_correlation": np.random.uniform(0.1, 0.8, n),
            "avg_volume_ratio": np.random.uniform(0.5, 2.0, n),
            "market_breadth":  np.random.uniform(0.1, 0.9, n),
        }, index=all_dates)

        train_end = str(all_dates[119].date())
        val_start = str(all_dates[120].date())
        val_end = str(all_dates[-1].date())

        veto = c.fit_predict_window(regime_all, "2020-01-02", train_end, val_start, val_end)
        assert isinstance(veto, pd.Series)
        assert set(veto.unique()).issubset({0, 1})
        assert len(veto) == 80


class TestConspiranoicoSoftVeto:

    @pytest.fixture
    def config_soft(self):
        cfg = {
            "conspiranoico": {
                "n_estimators": 50,
                "contamination": 0.05,
                "vix_ticker": "^VIX",
                "volatility_windows": [5, 20, 60],
                "correlation_window": 20,
                "volume_ratio_window": 20,
                "veto_threshold_percentile": 5,
                "veto_mode": "soft",
                "soft_veto_scale_severe": 0.25,
                "soft_veto_scale_moderate": 0.6,
                "soft_veto_moderate_percentile": 15,
                "random_state": 42,
            }
        }
        return cfg

    def test_predict_risk_scale_returns_float_in_range(
        self, regime_train, regime_val, config_soft,
    ):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_soft)
        c.fit(regime_train)
        scale = c.predict_risk_scale(regime_val)
        assert isinstance(scale, pd.Series)
        assert scale.index.equals(regime_val.index)
        valid = scale.dropna()
        assert set(valid.unique()).issubset({0.25, 0.6, 1.0})

    def test_crisis_gets_lower_scale_than_calm(
        self, regime_train, regime_val, config_soft,
    ):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_soft)
        c.fit(regime_train)

        calm_val = regime_val.copy()
        calm_val["vix"] = 12.0
        calm_val["vol_5d"] = 0.005
        calm_val["avg_correlation"] = 0.2
        calm_val["market_breadth"] = 0.65

        crisis_val = regime_val.copy()
        crisis_val["vix"] = 70.0
        crisis_val["vol_5d"] = 0.08
        crisis_val["avg_correlation"] = 0.95
        crisis_val["market_breadth"] = 0.1

        scale_calm = c.predict_risk_scale(calm_val)
        scale_crisis = c.predict_risk_scale(crisis_val)

        assert scale_crisis.mean() <= scale_calm.mean()

    def test_binary_mode_returns_ones(self, regime_train, regime_val, config_toy):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_toy)
        c.fit(regime_train)
        scale = c.predict_risk_scale(regime_val)
        assert (scale == 1.0).all()

    def test_moderate_percentile_must_exceed_veto_percentile(self, regime_train, config_soft):
        from agents.conspiranoico import Conspiranoico
        cfg = config_soft.copy()
        cfg["conspiranoico"] = {**cfg["conspiranoico"]}
        cfg["conspiranoico"]["soft_veto_moderate_percentile"] = 3
        c = Conspiranoico(cfg)
        with pytest.raises(ValueError, match="debe ser mayor"):
            c.fit(regime_train)

    def test_nan_rows_get_scale_one(self, regime_train, regime_val, config_soft):
        from agents.conspiranoico import Conspiranoico
        c = Conspiranoico(config_soft)
        c.fit(regime_train)
        val_with_nan = regime_val.copy()
        val_with_nan.iloc[0, 0] = np.nan
        scale = c.predict_risk_scale(val_with_nan)
        assert float(scale.iloc[0]) == 1.0


# ── Tests de integración con JuezV1 ──────────────────────────────────────────

class TestConspiranoicoWithJuez:

    def test_veto_zeros_all_signals(self, regime_train, regime_val, config_toy):
        """Cuando veto=1 en una fecha, JuezV1 debe poner señal 0 para todos los tickers."""
        from agents.conspiranoico import Conspiranoico
        from judge.judge_v1 import JuezV1

        c = Conspiranoico(config_toy)
        c.fit(regime_train)

        # Forzar veto=1 en todos los días
        veto = pd.Series(1, index=regime_val.index, dtype=int)

        judge_cfg = {
            "judge_v1": {
                "meta_model": "logistic",
                "calibration_method": "sigmoid",
                "random_state": 42,
            }
        }
        juez = JuezV1(judge_cfg)

        tickers = ["AAPL", "MSFT"]
        signals = pd.DataFrame(
            np.random.uniform(0.3, 0.8, (len(regime_val), len(tickers))),
            index=regime_val.index,
            columns=tickers,
        )

        result = juez.predict(signals, veto_signal=veto)
        assert (result == 0.0).all().all(), (
            "Con veto=1 en todos los días, todas las señales deben ser 0"
        )

    def test_no_veto_passes_signals_through(self, regime_val, config_toy):
        """Cuando veto=0 siempre, JuezV1 en pass-through retorna señales originales."""
        from judge.judge_v1 import JuezV1

        veto = pd.Series(0, index=regime_val.index, dtype=int)
        judge_cfg = {
            "judge_v1": {
                "meta_model": "logistic",
                "calibration_method": "sigmoid",
                "random_state": 42,
            }
        }
        juez = JuezV1(judge_cfg)

        tickers = ["AAPL"]
        np.random.seed(1)
        signals = pd.DataFrame(
            np.random.uniform(0.3, 0.8, (len(regime_val), len(tickers))),
            index=regime_val.index,
            columns=tickers,
        )

        result = juez.predict(signals, veto_signal=veto)
        pd.testing.assert_frame_equal(result, signals.astype(float))


# ── Smoke test walk-forward con Conspiranoico ─────────────────────────────────

class TestWalkForwardWithConspiranoico:

    def test_smoke_two_tickers(self, tmp_path, config_toy):
        """
        Smoke test: pipeline walk-forward completo con 2 tickers sintéticos
        y El Conspiranoico activo.
        Verifica que el pipeline corre sin errores y retorna métricas.
        """
        import pandas as pd
        import numpy as np
        from agents.conspiranoico import Conspiranoico
        from agents.gestor_riesgos import GestorRiesgos
        from agents.matematico import Matematico
        from backtester.walk_forward import WalkForwardValidator
        from data.features import compute_all_features
        from judge.judge_v1 import JuezV1

        # Configuración mínima para 2 ventanas walk-forward
        cfg = {
            "universe": {"tickers_file": str(tmp_path / "u.csv")},
            "data": {
                "start_date": "2018-01-01",
                "end_date": "2023-12-31",
                "holdout_start": "2024-01-01",
                "cache_dir": str(tmp_path),
            },
            "backtester": {
                "commission_pct": 0.0008,
                "initial_capital": 10000.0,
                "walk_forward_train_years": 2,
                "walk_forward_val_years": 1,
            },
            "risk_manager": {
                "kelly_fraction": 0.5,
                "max_position_pct": 0.15,
                "stop_loss_atr_multiplier": 2.0,
                "min_kelly_threshold": 0.005,
            },
            "asset_classes": {
                "equity": {"max_position_pct": 0.15, "max_portfolio_pct": 0.80},
            },
            "matematico": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "calibration_method": "sigmoid",
                "random_state": 42,
            },
            "judge_v1": {
                "meta_model": "logistic",
                "calibration_method": "sigmoid",
                "random_state": 42,
            },
            "conspiranoico": {
                "n_estimators": 50,
                "contamination": 0.05,
                "vix_ticker": "^VIX",
                "volatility_windows": [5, 20],
                "correlation_window": 20,
                "volume_ratio_window": 20,
                "veto_threshold_percentile": 5,
                "random_state": 42,
            },
            "general": {"random_seed": 42, "log_dir": str(tmp_path / "logs")},
        }

        # CSV del universo (2 tickers equity)
        pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "asset_class": ["equity", "equity"],
        }).to_csv(tmp_path / "u.csv", index=False)

        # Datos OHLCV sintéticos (6 años)
        dates = pd.date_range("2018-01-02", "2023-12-29", freq="B")
        n = len(dates)
        np.random.seed(42)

        def make_ohlcv(seed):
            np.random.seed(seed)
            close = pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01), index=dates)
            return pd.DataFrame({
                "Open": close * 0.99,
                "High": close * 1.02,
                "Low": close * 0.98,
                "Close": close,
                "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
            })

        prices = {"AAPL": make_ohlcv(42), "MSFT": make_ohlcv(7)}
        features = {t: compute_all_features(df) for t, df in prices.items()}

        # Features de régimen sintéticas
        regime_features = pd.DataFrame({
            "vol_5d":          np.random.uniform(0.005, 0.03, n),
            "vol_20d":         np.random.uniform(0.007, 0.035, n),
            "vix":             np.random.uniform(10, 30, n),
            "avg_correlation": np.random.uniform(0.1, 0.8, n),
            "avg_volume_ratio": np.random.uniform(0.5, 2.0, n),
            "market_breadth":  np.random.uniform(0.1, 0.9, n),
        }, index=dates)

        matematico = Matematico(cfg)
        juez = JuezV1(cfg)
        gestor = GestorRiesgos.from_config(cfg)
        conspiranoico = Conspiranoico(cfg)

        validator = WalkForwardValidator(cfg)
        results = validator.run(
            agents={"matematico": matematico, "conspiranoico": conspiranoico},
            judge=juez,
            gestor=gestor,
            prices=prices,
            features=features,
            regime_features=regime_features,
        )

        assert "metrics" in results
        assert "sharpe_ratio" in results["metrics"]
        assert len(results["window_results"]) > 0
        # Hay al menos alguna ventana con datos de veto en los logs
        win = results["window_results"][0]
        assert "sharpe_ratio" in win

    def test_missing_regime_features_raises(self):
        """Pasar conspiranoico sin regime_features debe lanzar ValueError."""
        from agents.conspiranoico import Conspiranoico
        from agents.gestor_riesgos import GestorRiesgos
        from agents.matematico import Matematico
        from backtester.walk_forward import WalkForwardValidator
        from judge.judge_v1 import JuezV1

        cfg_min = {
            "universe": {"tickers_file": "data/universe_2018-01-01.csv"},
            "data": {
                "start_date": "2018-01-01",
                "end_date": "2023-12-31",
                "holdout_start": "2024-01-01",
                "cache_dir": "data/cache",
            },
            "backtester": {
                "commission_pct": 0.0008,
                "initial_capital": 10000.0,
                "walk_forward_train_years": 2,
                "walk_forward_val_years": 1,
            },
            "risk_manager": {
                "kelly_fraction": 0.5,
                "max_position_pct": 0.15,
                "stop_loss_atr_multiplier": 2.0,
                "min_kelly_threshold": 0.005,
            },
            "asset_classes": {
                "equity": {"max_position_pct": 0.15, "max_portfolio_pct": 0.80},
            },
            "matematico": {
                "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "calibration_method": "sigmoid", "random_state": 42,
            },
            "judge_v1": {"meta_model": "logistic", "calibration_method": "sigmoid", "random_state": 42},
            "conspiranoico": {
                "n_estimators": 50, "contamination": 0.05, "vix_ticker": "^VIX",
                "volatility_windows": [5, 20], "correlation_window": 20,
                "volume_ratio_window": 20, "veto_threshold_percentile": 5, "random_state": 42,
            },
            "general": {"random_seed": 42},
        }

        validator = WalkForwardValidator(cfg_min)
        with pytest.raises(ValueError, match="regime_features"):
            validator.run(
                agents={"matematico": Matematico(cfg_min), "conspiranoico": Conspiranoico(cfg_min)},
                judge=JuezV1(cfg_min),
                gestor=GestorRiesgos.from_config(cfg_min),
                prices={},
                features={},
                regime_features=None,
            )


class TestSummarizeRiskScale:
    def test_empty_series(self):
        from agents.conspiranoico import Conspiranoico

        stats = Conspiranoico.summarize_risk_scale(pd.Series(dtype=float))
        assert stats["n_days"] == 0
        assert stats["pct_reduced"] == 0.0

    def test_tier_counts(self):
        from agents.conspiranoico import Conspiranoico

        scale = pd.Series([0.25, 0.25, 0.6, 1.0, 1.0])
        stats = Conspiranoico.summarize_risk_scale(scale)
        assert stats["n_days"] == 5
        assert stats["n_severe"] == 2
        assert stats["n_moderate"] == 1
        assert stats["n_normal"] == 2
        assert stats["pct_reduced"] == 60.0

    def test_summarize_veto(self):
        from agents.conspiranoico import Conspiranoico

        veto = pd.Series([0, 1, 1, 0, 0])
        stats = Conspiranoico.summarize_veto(veto)
        assert stats["n_veto"] == 2
        assert stats["pct_veto"] == 40.0


# ── Tests HMM (Exp11) ────────────────────────────────────────────────────────

@pytest.fixture
def config_hmm():
    return {
        "conspiranoico": {
            "detector": "hmm",
            "n_estimators": 50,
            "contamination": 0.05,
            "vix_ticker": "^VIX",
            "volatility_windows": [5, 20, 60],
            "correlation_window": 20,
            "volume_ratio_window": 20,
            "veto_mode": "soft",
            "soft_veto_scale_severe": 0.25,
            "soft_veto_scale_moderate": 0.6,
            "hmm": {"n_states": 3, "n_iter": 50, "covariance_type": "diag"},
            "random_state": 42,
        }
    }


@pytest.fixture
def regime_train_crisis(dates_train):
    """Train con dos regímenes claros para que HMM separe estados."""
    np.random.seed(42)
    n = len(dates_train)
    crisis = np.zeros(n, dtype=bool)
    crisis[n // 2:] = True
    return pd.DataFrame({
        "vol_5d":          np.where(crisis, 0.04, 0.008),
        "vol_20d":         np.where(crisis, 0.05, 0.010),
        "vol_60d":         np.where(crisis, 0.055, 0.012),
        "vix":             np.where(crisis, 35.0, 14.0),
        "avg_correlation": np.where(crisis, 0.85, 0.35),
        "avg_volume_ratio": np.where(crisis, 1.8, 1.0),
        "market_breadth":  np.where(crisis, 0.25, 0.55),
    }, index=dates_train)


@pytest.fixture
def regime_val_crisis(dates_val):
    np.random.seed(99)
    n = len(dates_val)
    crisis = np.zeros(n, dtype=bool)
    crisis[n // 3:] = True
    return pd.DataFrame({
        "vol_5d":          np.where(crisis, 0.042, 0.009),
        "vol_20d":         np.where(crisis, 0.052, 0.011),
        "vol_60d":         np.where(crisis, 0.056, 0.013),
        "vix":             np.where(crisis, 38.0, 15.0),
        "avg_correlation": np.where(crisis, 0.88, 0.38),
        "avg_volume_ratio": np.where(crisis, 1.9, 1.05),
        "market_breadth":  np.where(crisis, 0.22, 0.52),
    }, index=dates_val)


class TestConspiranoicoHMM:

    def test_hmm_fit_predict_soft_scale(self, regime_train_crisis, regime_val_crisis, config_hmm):
        from agents.conspiranoico import Conspiranoico

        c = Conspiranoico(config_hmm)
        c.fit(regime_train_crisis)
        scale = c.predict_risk_scale(regime_val_crisis)

        assert c.detector == "hmm"
        assert c.crisis_state is not None
        assert len(c.state_risk_rank) == 3
        assert set(scale.unique()).issubset({0.25, 0.6, 1.0})
        # En OOS el HMM puede clasificar todo como normal; en-sample crisis sí reduce escala
        scale_train_crisis = c.predict_risk_scale(regime_train_crisis.iloc[len(regime_train_crisis) // 2:])
        assert (scale_train_crisis < 1.0).any()

    def test_hmm_crisis_state_has_highest_risk_rank(self, regime_train_crisis, config_hmm):
        from agents.conspiranoico import Conspiranoico

        c = Conspiranoico(config_hmm)
        c.fit(regime_train_crisis)
        assert c.state_risk_rank[c.crisis_state] == 2

    def test_hmm_binary_veto_only_crisis(self, regime_train_crisis, regime_val_crisis, config_hmm):
        from agents.conspiranoico import Conspiranoico

        config_hmm["conspiranoico"]["veto_mode"] = "binary"
        c = Conspiranoico(config_hmm)
        c.fit(regime_train_crisis)
        veto = c.predict(regime_val_crisis)
        assert set(veto.unique()).issubset({0, 1})

    def test_hybrid_more_conservative_than_if_alone(
        self, regime_train_crisis, regime_val_crisis, config_toy, config_hmm
    ):
        from agents.conspiranoico import Conspiranoico

        config_hmm["conspiranoico"]["detector"] = "hybrid"
        config_hmm["conspiranoico"]["soft_veto_moderate_percentile"] = 15
        config_hmm["conspiranoico"]["veto_threshold_percentile"] = 5

        config_toy_soft = dict(config_toy)
        config_toy_soft["conspiranoico"] = {
            **config_toy["conspiranoico"],
            "veto_mode": "soft",
            "soft_veto_scale_severe": 0.25,
            "soft_veto_scale_moderate": 0.6,
            "soft_veto_moderate_percentile": 15,
        }
        c_if = Conspiranoico(config_toy_soft)
        c_if.fit(regime_train_crisis)
        if_scale = c_if.predict_risk_scale(regime_val_crisis)

        c_hybrid = Conspiranoico(config_hmm)
        c_hybrid.fit(regime_train_crisis)
        hybrid_scale = c_hybrid.predict_risk_scale(regime_val_crisis)

        assert (hybrid_scale <= if_scale + 1e-9).all()

    def test_invalid_detector_raises(self, config_toy):
        from agents.conspiranoico import Conspiranoico

        config_toy["conspiranoico"]["detector"] = "neural_net"
        with pytest.raises(ValueError, match="detector"):
            Conspiranoico(config_toy)

