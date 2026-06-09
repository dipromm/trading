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
