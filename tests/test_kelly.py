"""
Tests del Gestor de Riesgos (Fractional Kelly).

Estos tests deben pasar desde el momento en que se implementa GestorRiesgos.
Son la garantía de que el sistema de posicionamiento es matemáticamente correcto.
"""

import pytest

from agents.gestor_riesgos import GestorRiesgos


@pytest.fixture
def gestor():
    return GestorRiesgos(
        kelly_fraction=0.5,
        max_position_pct=0.15,
        stop_loss_atr_multiplier=2.0,
        min_kelly_threshold=0.005,
    )


class TestComputePositionSize:

    def test_negative_kelly_returns_zero(self, gestor):
        """f* <= 0 nunca debe generar una posición de compra."""
        # p=0.3, b=1.0 → kelly_full = 0.3 - 0.7/1.0 = -0.4 → f* = -0.2 → 0.0
        f_star = gestor.compute_position_size(p=0.3, b=1.0)
        assert f_star == 0.0, f"Kelly negativo retornó {f_star}, esperaba 0.0"

    def test_cap_at_max_position(self, gestor):
        """f* nunca debe superar el cap del 15%."""
        # p=0.95, b=10.0 → kelly muy alto → debe capar en 0.15
        f_star = gestor.compute_position_size(p=0.95, b=10.0)
        assert f_star <= 0.15, f"Posición {f_star:.4f} supera el cap del 15%"

    def test_half_kelly_applied(self, gestor):
        """Verifica que se aplica el factor rho=0.5 correctamente."""
        # p=0.6, b=1.0 → kelly_full = 0.6 - 0.4/1.0 = 0.2 → f* = 0.5 × 0.2 = 0.1
        f_star = gestor.compute_position_size(p=0.6, b=1.0)
        assert abs(f_star - 0.1) < 1e-9, f"Half-Kelly esperaba 0.1, obtenido {f_star}"

    def test_kelly_fraction_override(self, gestor):
        """Override de rho para Kelly dinámico (Exp5)."""
        f_base = gestor.compute_position_size(p=0.6, b=1.0)
        f_boost = gestor.compute_position_size(p=0.6, b=1.0, kelly_fraction=0.6)
        assert abs(f_boost - f_base * 1.2) < 1e-9

    def test_zero_b_returns_zero(self, gestor):
        """b=0 no debe causar ZeroDivisionError."""
        f_star = gestor.compute_position_size(p=0.7, b=0.0)
        assert f_star == 0.0

    def test_negative_b_returns_zero(self, gestor):
        """b negativo no tiene sentido financiero → retornar 0."""
        f_star = gestor.compute_position_size(p=0.7, b=-1.0)
        assert f_star == 0.0

    def test_p_exactly_zero(self, gestor):
        """p=0 → kelly muy negativo → f* = 0."""
        f_star = gestor.compute_position_size(p=0.0, b=1.0)
        assert f_star == 0.0

    def test_p_exactly_one(self, gestor):
        """p=1.0 → kelly máximo → f* capado al 15%."""
        f_star = gestor.compute_position_size(p=1.0, b=1.0)
        assert f_star <= 0.15

    def test_result_in_valid_range(self, gestor):
        """El resultado siempre debe estar en [0, max_position_pct]."""
        test_cases = [
            (0.4, 0.5), (0.5, 1.0), (0.6, 1.5), (0.7, 2.0), (0.8, 3.0),
        ]
        for p, b in test_cases:
            f_star = gestor.compute_position_size(p=p, b=b)
            assert 0.0 <= f_star <= 0.15, \
                f"f*={f_star:.4f} fuera de rango [0, 0.15] para p={p}, b={b}"


class TestNormalizePortfolio:

    def test_no_normalization_needed(self, gestor):
        """Si la suma <= 1.0, no se modifica nada."""
        positions = {"AAPL": 0.10, "MSFT": 0.10, "NVDA": 0.05}
        result = gestor.normalize_portfolio(positions)
        assert result == positions

    def test_normalization_when_over_100(self, gestor):
        """Si la suma > 1.0, la suma normalizada debe ser <= 1.0."""
        positions = {f"TICK{i}": 0.14 for i in range(8)}  # 8 × 0.14 = 1.12 > 1.0
        result = gestor.normalize_portfolio(positions)
        total = sum(result.values())
        assert total <= 1.0 + 1e-9, f"Suma normalizada {total:.4f} supera 1.0"

    def test_cap_maintained_after_normalization(self, gestor):
        """Después de normalizar, ningún ticker supera el cap del 15%."""
        positions = {f"TICK{i}": 0.14 for i in range(8)}
        result = gestor.normalize_portfolio(positions)
        for ticker, pos in result.items():
            assert pos <= 0.15 + 1e-9, f"{ticker}: posición {pos:.4f} supera el cap"

    def test_empty_portfolio(self, gestor):
        """Portfolio vacío no debe causar errores."""
        result = gestor.normalize_portfolio({})
        assert result == {}


class TestStopLoss:

    def test_stop_loss_triggered(self, gestor):
        """Stop-loss debe activarse cuando la caída supera N × ATR."""
        # entry=100, atr=2.0, multiplier=2.0 → stop=96. precio=95 < 96 → activar
        assert gestor.should_stop_loss(current_price=95.0, entry_price=100.0, atr=2.0)

    def test_stop_loss_not_triggered(self, gestor):
        """Stop-loss NO debe activarse si la caída es menor que N × ATR."""
        # entry=100, atr=2.0, multiplier=2.0 → stop=96. precio=97 > 96 → no activar
        assert not gestor.should_stop_loss(current_price=97.0, entry_price=100.0, atr=2.0)

    def test_stop_loss_at_exact_level(self, gestor):
        """En el nivel exacto del stop-loss se debe activar."""
        # entry=100, atr=2.0, stop=96. precio=96 → activar
        assert gestor.should_stop_loss(current_price=96.0, entry_price=100.0, atr=2.0)


class TestAssetClassCaps:
    """Tests de caps diferenciados por clase de activo."""

    @pytest.fixture
    def gestor_multi(self):
        return GestorRiesgos(
            kelly_fraction=0.5,
            max_position_pct=0.15,
            stop_loss_atr_multiplier=2.0,
            min_kelly_threshold=0.005,
            asset_position_caps={
                "equity": 0.15,
                "bond": 0.20,
                "gold": 0.10,
                "defensive_equity": 0.15,
                "international_equity": 0.10,
            },
            asset_portfolio_caps={
                "equity": 0.80,
                "bond": 0.30,
                "gold": 0.10,
                "defensive_equity": 0.20,
                "international_equity": 0.10,
            },
        )

    def test_bond_cap_higher_than_equity(self, gestor_multi):
        """Los bonos tienen cap por ticker mayor que las acciones (20% vs 15%)."""
        assert gestor_multi.get_position_cap("bond") > gestor_multi.get_position_cap("equity")

    def test_gold_cap_lower_than_equity(self, gestor_multi):
        """El oro tiene cap por ticker menor que las acciones (10% vs 15%)."""
        assert gestor_multi.get_position_cap("gold") < gestor_multi.get_position_cap("equity")

    def test_bond_position_respects_bond_cap(self, gestor_multi):
        """f* de un bono no supera el 20%."""
        # p=0.95, b=10.0 → kelly muy alto → debe capar en 0.20 (cap de bonos)
        f_star = gestor_multi.compute_position_size(p=0.95, b=10.0, asset_class="bond")
        assert f_star <= 0.20, f"Posición de bono {f_star:.4f} supera cap del 20%"

    def test_gold_position_respects_gold_cap(self, gestor_multi):
        """f* del oro no supera el 10%."""
        f_star = gestor_multi.compute_position_size(p=0.95, b=10.0, asset_class="gold")
        assert f_star <= 0.10, f"Posición de oro {f_star:.4f} supera cap del 10%"

    def test_unknown_class_uses_default_cap(self, gestor_multi):
        """Clase de activo desconocida usa el cap por defecto."""
        f_star = gestor_multi.compute_position_size(p=0.95, b=10.0, asset_class="unknown")
        assert f_star <= gestor_multi.max_position_pct

    def test_portfolio_class_cap_bonds_respected(self, gestor_multi):
        """La suma de bonos en el portfolio no supera el 30%."""
        positions = {"TLT": 0.20, "IEF": 0.20}  # 40% > 30% cap de bonos
        ticker_classes = {"TLT": "bond", "IEF": "bond"}

        result = gestor_multi.normalize_portfolio_by_class(positions, ticker_classes)
        bond_total = sum(result[t] for t in ["TLT", "IEF"])
        assert bond_total <= 0.30 + 1e-9, \
            f"Total bonos {bond_total:.4f} supera el cap del portfolio del 30%"

    def test_portfolio_class_cap_equity_respected(self, gestor_multi):
        """La suma de equity en el portfolio no supera el 80%."""
        positions = {f"TICK{i}": 0.10 for i in range(10)}  # 10 × 10% = 100% > 80%
        ticker_classes = {f"TICK{i}": "equity" for i in range(10)}

        result = gestor_multi.normalize_portfolio_by_class(positions, ticker_classes)
        equity_total = sum(result.values())
        assert equity_total <= 0.80 + 1e-9, \
            f"Total equity {equity_total:.4f} supera el cap del portfolio del 80%"

    def test_mixed_portfolio_total_under_100(self, gestor_multi):
        """Portfolio mixto (acciones + bonos + oro) no supera el 100%."""
        positions = {
            "AAPL": 0.14, "MSFT": 0.14, "NVDA": 0.14,   # equity
            "TLT": 0.18, "IEF": 0.18,                     # bond (36% > cap 30%)
            "GLD": 0.09,                                    # gold
        }
        ticker_classes = {
            "AAPL": "equity", "MSFT": "equity", "NVDA": "equity",
            "TLT": "bond", "IEF": "bond",
            "GLD": "gold",
        }
        result = gestor_multi.normalize_portfolio_by_class(positions, ticker_classes)
        total = sum(result.values())
        assert total <= 1.0 + 1e-9, f"Portfolio total {total:.4f} supera el 100%"

    def test_existing_tests_still_pass_without_classes(self, gestor_multi):
        """normalize_portfolio() original sigue funcionando para backward compat."""
        positions = {"AAPL": 0.14, "MSFT": 0.14, "NVDA": 0.14}
        result = gestor_multi.normalize_portfolio(positions)
        assert sum(result.values()) <= 1.0 + 1e-9
