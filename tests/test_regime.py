"""
Tests para data/regime.py — features de régimen de mercado.

Verificaciones críticas:
    1. Anti-leakage: las features del día T no usan datos de T+1.
    2. market_breadth es correcto para datos toy conocidos.
    3. avg_correlation se comporta correctamente en casos extremos.
    4. build_regime_features produce las columnas esperadas.
    5. VIX se alinea correctamente al índice de fechas del universo.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── Fixtures comunes ─────────────────────────────────────────────────────────

@pytest.fixture
def dates():
    return pd.date_range("2021-01-04", periods=100, freq="B")  # días hábiles


@pytest.fixture
def prices_toy(dates):
    """Dos acciones equity con precios sintéticos controlados."""
    np.random.seed(42)
    n = len(dates)
    prices = {}
    for ticker in ["AAPL", "MSFT"]:
        close = pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01), index=dates)
        volume = pd.Series(np.random.randint(1_000_000, 5_000_000, n), index=dates)
        df = pd.DataFrame({
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": volume.astype(float),
        })
        prices[ticker] = df
    return prices


@pytest.fixture
def vix_toy(dates):
    """VIX sintético con valores plausibles."""
    np.random.seed(7)
    values = 15 + np.random.rand(len(dates)) * 10
    return pd.Series(values, index=dates, name="Close")


@pytest.fixture
def config_toy(tmp_path):
    """Config mínimo que apunta a un CSV temporal de universo."""
    csv_path = tmp_path / "universe.csv"
    pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "asset_class": ["equity", "equity"],
    }).to_csv(csv_path, index=False)

    return {
        "universe": {"tickers_file": str(csv_path)},
        "data": {
            "start_date": "2021-01-01",
            "end_date": "2022-01-01",
            "cache_dir": str(tmp_path),
        },
        "conspiranoico": {
            "n_estimators": 100,
            "contamination": 0.05,
            "vix_ticker": "^VIX",
            "volatility_windows": [5, 20, 60],
            "correlation_window": 20,
            "volume_ratio_window": 20,
            "random_state": 42,
        },
    }


# ── Tests de build_returns_matrix ────────────────────────────────────────────

class TestBuildReturnsMatrix:

    def test_returns_shape(self, prices_toy, dates):
        from data.regime import build_returns_matrix
        returns = build_returns_matrix(prices_toy)
        assert set(returns.columns) == {"AAPL", "MSFT"}
        assert len(returns) == len(dates)

    def test_first_row_is_nan(self, prices_toy):
        """El retorno del primer día siempre es NaN (no hay T-1)."""
        from data.regime import build_returns_matrix
        returns = build_returns_matrix(prices_toy)
        assert returns.iloc[0].isna().all()

    def test_returns_lookback_safe(self, prices_toy, dates):
        """
        Anti-leakage: insertar un shock en T no debe modificar los retornos de T-1.
        """
        from data.regime import build_returns_matrix

        returns_before = build_returns_matrix(prices_toy)

        # Duplicar prices y alterar el precio de un día en medio
        import copy
        prices_shocked = copy.deepcopy(prices_toy)
        shock_idx = 50
        prices_shocked["AAPL"].iloc[shock_idx, prices_shocked["AAPL"].columns.get_loc("Close")] *= 2.0

        returns_after = build_returns_matrix(prices_shocked)

        # El retorno del día anterior al shock NO debe cambiar
        assert returns_before["AAPL"].iloc[shock_idx - 1] == pytest.approx(
            returns_after["AAPL"].iloc[shock_idx - 1], abs=1e-10
        )


# ── Tests de compute_market_breadth ──────────────────────────────────────────

class TestMarketBreadth:

    def test_all_positive(self, dates):
        from data.regime import compute_market_breadth
        n = 50
        dates_slice = dates[:n]
        returns = pd.DataFrame({
            "A": [0.01] * n,
            "B": [0.02] * n,
            "C": [0.005] * n,
        }, index=dates_slice)
        breadth = compute_market_breadth(returns)
        valid = breadth.dropna().values
        assert np.allclose(valid, 1.0)

    def test_all_negative(self, dates):
        from data.regime import compute_market_breadth
        n = 50
        dates_slice = dates[:n]
        returns = pd.DataFrame({
            "A": [-0.01] * n,
            "B": [-0.02] * n,
            "C": [-0.005] * n,
        }, index=dates_slice)
        breadth = compute_market_breadth(returns)
        valid = breadth.dropna().values
        assert np.allclose(valid, 0.0)

    def test_half_positive(self, dates):
        from data.regime import compute_market_breadth
        n = 50
        dates_slice = dates[:n]
        returns = pd.DataFrame({
            "A": [0.01] * n,
            "B": [-0.01] * n,
        }, index=dates_slice)
        breadth = compute_market_breadth(returns)
        valid = breadth.dropna().values
        assert np.allclose(valid, 0.5)

    def test_range_is_zero_to_one(self, prices_toy, dates):
        from data.regime import build_returns_matrix, compute_market_breadth
        returns = build_returns_matrix(prices_toy)
        breadth = compute_market_breadth(returns)
        valid = breadth.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()


# ── Tests de compute_rolling_correlation ─────────────────────────────────────

class TestRollingCorrelation:

    def test_perfectly_correlated(self, dates):
        """Dos series idénticas → correlación = 1.0."""
        from data.regime import compute_rolling_correlation
        n = 60
        series = pd.Series(np.random.randn(n), index=dates[:n])
        returns = pd.DataFrame({"A": series, "B": series})
        corr = compute_rolling_correlation(returns, window=20)
        valid = corr.dropna()
        assert len(valid) > 0
        assert (valid > 0.99).all()

    def test_perfectly_anticorrelated(self, dates):
        """Dos series opuestas → correlación = -1.0."""
        from data.regime import compute_rolling_correlation
        n = 60
        series = pd.Series(np.random.randn(n), index=dates[:n])
        returns = pd.DataFrame({"A": series, "B": -series})
        corr = compute_rolling_correlation(returns, window=20)
        valid = corr.dropna()
        assert len(valid) > 0
        assert (valid < -0.99).all()

    def test_single_ticker_returns_nan(self, dates):
        """Con un solo ticker no se puede calcular correlación pairwise."""
        from data.regime import compute_rolling_correlation
        n = 60
        returns = pd.DataFrame({"A": np.random.randn(n)}, index=dates[:n])
        corr = compute_rolling_correlation(returns, window=20)
        assert corr.isna().all()


# ── Tests de build_regime_features ───────────────────────────────────────────

class TestBuildRegimeFeatures:

    def test_columns_present(self, prices_toy, vix_toy, config_toy):
        from data.regime import build_regime_features
        features = build_regime_features(prices_toy, vix_toy, config_toy)
        expected_cols = {
            "vol_5d", "vol_20d", "vol_60d",
            "vix",
            "avg_correlation",
            "avg_volume_ratio",
            "market_breadth",
        }
        assert expected_cols.issubset(set(features.columns)), (
            f"Columnas faltantes: {expected_cols - set(features.columns)}"
        )

    def test_no_future_data_in_features(self, prices_toy, vix_toy, config_toy):
        """
        Anti-leakage: añadir un día extra con precios inflados no debe
        modificar las features de los días anteriores.
        """
        from data.regime import build_regime_features
        import copy

        features_orig = build_regime_features(prices_toy, vix_toy, config_toy)

        # Añadir un día extra con precios 10× en prices_toy
        extra_date = prices_toy["AAPL"].index[-1] + pd.Timedelta(days=1)
        prices_shock = copy.deepcopy(prices_toy)
        for ticker in prices_shock:
            last_row = prices_shock[ticker].iloc[[-1]].copy()
            last_row.index = [extra_date]
            last_row["Close"] = last_row["Close"] * 10
            prices_shock[ticker] = pd.concat([prices_shock[ticker], last_row])

        extra_vix = pd.concat([vix_toy, pd.Series([100.0], index=[extra_date])])
        features_new = build_regime_features(prices_shock, extra_vix, config_toy)

        # El penúltimo día debe ser idéntico en ambos
        common_dates = features_orig.index.intersection(features_new.index)
        last_common = common_dates[-1]
        pd.testing.assert_series_equal(
            features_orig.loc[last_common],
            features_new.loc[last_common],
            check_names=False,
            rtol=1e-6,
        )

    def test_vix_alignment(self, prices_toy, config_toy, dates):
        """VIX con fechas distintas se alinea correctamente por forward-fill."""
        from data.regime import build_regime_features

        # VIX con algunas fechas faltantes
        vix_sparse = pd.Series(
            [20.0, 25.0, 18.0],
            index=[dates[0], dates[10], dates[50]],
        )
        features = build_regime_features(prices_toy, vix_sparse, config_toy)
        # Después de forward-fill, el día 11 debe tener vix=25
        assert not np.isnan(features["vix"].iloc[11])

    def test_output_index_matches_returns(self, prices_toy, vix_toy, config_toy):
        """El índice de las features debe coincidir con el índice de los precios."""
        from data.regime import build_regime_features, build_returns_matrix
        returns = build_returns_matrix(prices_toy)
        features = build_regime_features(prices_toy, vix_toy, config_toy)
        assert features.index.equals(returns.index)

    def test_no_future_shift_in_source(self):
        """
        Anti-leakage: verificar que data/regime.py no contiene llamadas .shift(-N)
        (método pandas real) fuera de comentarios o docstrings.
        """
        import inspect
        from data import regime
        source = inspect.getsource(regime)
        lines = source.split("\n")
        # Buscar el patrón de llamada de método real: ".shift(-" con punto delante
        leaky = [
            line for line in lines
            if ".shift(-" in line and not line.lstrip().startswith("#")
        ]
        assert len(leaky) == 0, (
            "Posible data leakage en data/regime.py: .shift(-N) encontrado:\n"
            + "\n".join(leaky)
        )


class TestKellyBoost:

    def test_boost_on_low_vix_normal_regime(self, dates):
        from data.regime import compute_kelly_fraction_by_date

        train = pd.DataFrame({"vix": np.linspace(10, 30, 50)}, index=dates[:50])
        val = pd.DataFrame({"vix": [12.0, 25.0, 15.0]}, index=dates[50:53])
        val_index = val.index
        risk_scale = pd.Series(1.0, index=val_index)
        veto = pd.Series(0, index=val_index)

        kelly, thr = compute_kelly_fraction_by_date(
            val_index=val_index,
            train_regime=train,
            val_regime=val,
            base_fraction=0.5,
            boost_fraction=0.6,
            vix_percentile=70,
            risk_scale=risk_scale,
            veto=veto,
        )

        assert thr == pytest.approx(float(np.percentile(train["vix"], 70)))
        assert kelly.iloc[0] == 0.6
        assert kelly.iloc[1] == 0.5
        assert kelly.iloc[2] == 0.6

    def test_no_boost_when_veto_active(self, dates):
        from data.regime import compute_kelly_fraction_by_date

        train = pd.DataFrame({"vix": [15.0] * 20}, index=dates[:20])
        val = pd.DataFrame({"vix": [14.0]}, index=dates[20:21])
        val_index = val.index

        kelly, _ = compute_kelly_fraction_by_date(
            val_index=val_index,
            train_regime=train,
            val_regime=val,
            base_fraction=0.5,
            boost_fraction=0.6,
            vix_percentile=70,
            risk_scale=pd.Series(1.0, index=val_index),
            veto=pd.Series(1, index=val_index),
        )
        assert kelly.iloc[0] == 0.5

    def test_no_boost_when_risk_scale_reduced(self, dates):
        from data.regime import compute_kelly_fraction_by_date

        train = pd.DataFrame({"vix": [15.0] * 20}, index=dates[:20])
        val = pd.DataFrame({"vix": [14.0]}, index=dates[20:21])
        val_index = val.index

        kelly, _ = compute_kelly_fraction_by_date(
            val_index=val_index,
            train_regime=train,
            val_regime=val,
            base_fraction=0.5,
            boost_fraction=0.6,
            vix_percentile=70,
            risk_scale=pd.Series(0.6, index=val_index),
            veto=pd.Series(0, index=val_index),
        )
        assert kelly.iloc[0] == 0.5


class TestDownloadVix:

    def test_uses_cache_without_network(self, config_toy, tmp_path):
        from data.regime import download_vix

        cache_file = tmp_path / "vix_index.parquet"
        dates = pd.date_range("2021-01-04", periods=5, freq="B")
        pd.DataFrame({"Close": [18.0, 19.0, 20.0, 21.0, 22.0]}, index=dates).to_parquet(cache_file)

        series = download_vix(config_toy)
        assert len(series) == 5
        assert series.iloc[-1] == 22.0

    def test_history_fallback_when_download_empty(self, config_toy, monkeypatch):
        from data.regime import download_vix

        dates = pd.date_range("2021-01-04", periods=3, freq="B")
        ohlcv = pd.DataFrame(
            {"Open": [18.0, 19.0, 20.0], "Close": [18.5, 19.5, 20.5]},
            index=dates,
        )

        class FakeTicker:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

            def history(self, *args, **kwargs):
                if kwargs.get("period") == "max" or args or kwargs.get("start"):
                    return ohlcv.copy()
                return pd.DataFrame()

        def fake_download(*args, **kwargs):
            return pd.DataFrame()

        import yfinance as yf

        monkeypatch.setattr(yf, "download", fake_download)
        monkeypatch.setattr(yf, "Ticker", FakeTicker)

        series = download_vix(config_toy, force_download=True)
        assert len(series) == 3
        assert (Path(config_toy["data"]["cache_dir"]) / "vix_index.parquet").exists()

    def test_stale_cache_when_all_downloads_fail(self, config_toy, monkeypatch):
        from data.regime import download_vix

        cache_file = Path(config_toy["data"]["cache_dir"]) / "vix_index.parquet"
        dates = pd.date_range("2021-01-04", periods=2, freq="B")
        pd.DataFrame({"Close": [17.0, 18.0]}, index=dates).to_parquet(cache_file)

        import yfinance as yf

        monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())

        class EmptyTicker:
            def __init__(self, symbol: str) -> None:
                pass

            def history(self, *args, **kwargs):
                return pd.DataFrame()

        monkeypatch.setattr(yf, "Ticker", EmptyTicker)

        series = download_vix(config_toy, force_download=True)
        assert len(series) == 2
        assert series.iloc[-1] == 18.0
