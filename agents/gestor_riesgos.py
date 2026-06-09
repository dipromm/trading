"""
El Gestor de Riesgos — Fractional Kelly + Stop-Loss dinámico + Diversificación.

No predice nada. Recibe las probabilidades calibradas del Juez y calcula:
  1. El tamaño de posición por ticker via Fractional Kelly (Half-Kelly)
  2. Caps diferenciados por clase de activo (acciones, bonos, oro, defensivos, internacional)
  3. Límites de concentración a nivel de clase de activo en el portfolio global
  4. Si una posición existente debe cerrarse por stop-loss (N × ATR)
  5. La normalización del portfolio respetando caps por ticker y por clase

Cadena de decisión:
    señal del Juez (p calibrada) + clase de activo del ticker
        → compute_position_size(p, b, asset_class) → f* bruto
        → min(f*, get_position_cap(asset_class)) → f* con cap por clase
        → normalize_portfolio_by_class() → respeta caps de clase + total 100%
        → orden final
"""

import logging
from typing import Optional

import pandas as pd

from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Caps por defecto si no se especifica config (respaldo)
_DEFAULT_ASSET_POSITION_CAPS: dict[str, float] = {
    "equity": 0.15,
    "bond": 0.20,
    "gold": 0.10,
    "defensive_equity": 0.15,
    "international_equity": 0.10,
}

_DEFAULT_ASSET_PORTFOLIO_CAPS: dict[str, float] = {
    "equity": 0.80,
    "bond": 0.30,
    "gold": 0.10,
    "defensive_equity": 0.20,
    "international_equity": 0.10,
}


class GestorRiesgos:
    """
    Gestor de riesgos basado en Fractional Kelly con límites de concentración
    diferenciados por clase de activo.

    Args:
        kelly_fraction: rho — factor de escala del Kelly puro (0.5 = Half-Kelly)
        max_position_pct: Cap por defecto si la clase de activo no está configurada
        stop_loss_atr_multiplier: N en stop-loss = N × ATR
        min_kelly_threshold: Posiciones menores a este valor se ignoran (ruido)
        asset_position_caps: Caps por ticker según clase de activo
        asset_portfolio_caps: Caps máximos del portfolio total por clase de activo
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.15,
        stop_loss_atr_multiplier: float = 2.0,
        min_kelly_threshold: float = 0.005,
        asset_position_caps: Optional[dict[str, float]] = None,
        asset_portfolio_caps: Optional[dict[str, float]] = None,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.min_kelly_threshold = min_kelly_threshold
        self.asset_position_caps = asset_position_caps or _DEFAULT_ASSET_POSITION_CAPS.copy()
        self.asset_portfolio_caps = asset_portfolio_caps or _DEFAULT_ASSET_PORTFOLIO_CAPS.copy()

    @classmethod
    def from_config(cls, config: dict | None = None) -> "GestorRiesgos":
        """Instancia el Gestor de Riesgos con los parámetros de config.yaml."""
        if config is None:
            config = load_config()
        rm = config["risk_manager"]

        asset_classes = config.get("asset_classes", {})
        asset_position_caps = {
            cls_name: cls_cfg["max_position_pct"]
            for cls_name, cls_cfg in asset_classes.items()
        } if asset_classes else _DEFAULT_ASSET_POSITION_CAPS.copy()

        asset_portfolio_caps = {
            cls_name: cls_cfg["max_portfolio_pct"]
            for cls_name, cls_cfg in asset_classes.items()
        } if asset_classes else _DEFAULT_ASSET_PORTFOLIO_CAPS.copy()

        return cls(
            kelly_fraction=rm["kelly_fraction"],
            max_position_pct=rm["max_position_pct"],
            stop_loss_atr_multiplier=rm["stop_loss_atr_multiplier"],
            min_kelly_threshold=rm["min_kelly_threshold"],
            asset_position_caps=asset_position_caps,
            asset_portfolio_caps=asset_portfolio_caps,
        )

    # ── Caps ──────────────────────────────────────────────────────────────────

    def get_position_cap(self, asset_class: str = "equity") -> float:
        """Retorna el cap de posición máxima por ticker para una clase de activo."""
        return self.asset_position_caps.get(asset_class, self.max_position_pct)

    def get_portfolio_cap(self, asset_class: str = "equity") -> float:
        """Retorna el cap máximo del portfolio total para una clase de activo."""
        return self.asset_portfolio_caps.get(asset_class, 1.0)

    # ── Kelly ─────────────────────────────────────────────────────────────────

    def compute_position_size(
        self,
        p: float,
        b: float,
        asset_class: str = "equity",
    ) -> float:
        """
        Calcula la fracción de capital a invertir via Fractional Kelly.

        Fórmula:
            f* = rho × (p - (1 - p) / b)

        Donde:
            p   = probabilidad calibrada de subida [0, 1]
            b   = ratio ganancia media / pérdida media histórico (> 0)
            rho = self.kelly_fraction (Half-Kelly = 0.5)

        Reglas:
            - Si b <= 0: retorna 0.0 (indefinido)
            - Si f* <= 0: retorna 0.0 (sin edge o señal bajista → no operar)
            - Si f* > cap de la clase: retorna el cap (duro)
            - Si f* < min_kelly_threshold: retorna 0.0 (ruido)

        Args:
            p: Probabilidad calibrada de subida [0, 1]
            b: Ratio ganancia media / pérdida media histórico.
               Calcular con datos del período de entrenamiento walk-forward actual.
            asset_class: Clase de activo del ticker. Determina el cap máximo.
                         Valores: "equity", "bond", "gold", "defensive_equity",
                         "international_equity"

        Returns:
            f* ∈ [0, get_position_cap(asset_class)]
        """
        if b <= 0:
            return 0.0

        kelly_full = p - (1 - p) / b
        f_star = self.kelly_fraction * kelly_full

        if f_star <= 0:
            return 0.0

        if f_star < self.min_kelly_threshold:
            return 0.0

        cap = self.get_position_cap(asset_class)
        return min(f_star, cap)

    # ── Gestión de portfolio ──────────────────────────────────────────────────

    def normalize_portfolio(self, positions: dict[str, float]) -> dict[str, float]:
        """
        Normalización simple: si la suma supera 1.0, normaliza proporcionalmente.
        Aplica el cap global max_position_pct para todos los tickers.

        Mantenido por compatibilidad con tests existentes.
        Para cartera con múltiples clases de activo usar normalize_portfolio_by_class().
        """
        total = sum(positions.values())
        if total <= 1.0:
            return positions

        normalized = {ticker: pos / total for ticker, pos in positions.items()}
        return {
            ticker: min(pos, self.max_position_pct)
            for ticker, pos in normalized.items()
        }

    def normalize_portfolio_by_class(
        self,
        positions: dict[str, float],
        ticker_classes: dict[str, str],
    ) -> dict[str, float]:
        """
        Normalización completa con tres niveles de restricción:

        Nivel 1 — Cap por ticker:
            Ningún ticker supera get_position_cap(su_clase).

        Nivel 2 — Cap por clase de activo:
            La suma de posiciones de cada clase no supera get_portfolio_cap(clase).
            Si la supera, se reducen proporcionalmente los tickers de esa clase.

        Nivel 3 — Cap total del portfolio:
            La suma total no supera 1.0 (100% del capital).
            Si la supera, se reduce todo proporcionalmente.

        Args:
            positions: dict {ticker: f*} resultado de compute_position_size()
            ticker_classes: dict {ticker: asset_class}

        Returns:
            dict {ticker: fracción final} con todos los caps respetados.
        """
        result = dict(positions)

        # Nivel 1: cap por ticker según su clase de activo
        for ticker, pos in result.items():
            asset_class = ticker_classes.get(ticker, "equity")
            cap = self.get_position_cap(asset_class)
            result[ticker] = min(pos, cap)

        # Nivel 2: cap por clase de activo en el portfolio total
        classes_present = set(ticker_classes.values())
        for asset_class in classes_present:
            class_tickers = [t for t, c in ticker_classes.items() if c == asset_class and t in result]
            class_total = sum(result[t] for t in class_tickers)
            portfolio_cap = self.get_portfolio_cap(asset_class)

            if class_total > portfolio_cap and class_total > 0:
                scale = portfolio_cap / class_total
                for ticker in class_tickers:
                    result[ticker] *= scale
                logger.debug(
                    f"Clase '{asset_class}': total {class_total:.3f} > cap {portfolio_cap:.3f}. "
                    f"Escalado ×{scale:.3f}"
                )

        # Nivel 3: cap total del portfolio (suma ≤ 1.0)
        total = sum(result.values())
        if total > 1.0:
            scale = 1.0 / total
            result = {ticker: pos * scale for ticker, pos in result.items()}
            logger.debug(f"Portfolio total {total:.3f} > 1.0. Escalado ×{scale:.3f}")

        return result

    # ── Stop-Loss ─────────────────────────────────────────────────────────────

    def should_stop_loss(
        self,
        current_price: float,
        entry_price: float,
        atr: float,
    ) -> bool:
        """
        Evalúa si se debe activar el stop-loss dinámico.

        El stop-loss se activa cuando el precio cae más de N × ATR
        por debajo del precio de entrada.

        Stop-loss = entry_price - (N × ATR)

        Args:
            current_price: Precio de cierre actual
            entry_price: Precio al que se abrió la posición
            atr: ATR del día actual (calculado en data/features.py)

        Returns:
            True si el stop-loss se ha activado (cerrar posición)
        """
        stop_level = entry_price - self.stop_loss_atr_multiplier * atr
        triggered = current_price <= stop_level
        if triggered:
            logger.debug(
                f"Stop-loss activado: precio={current_price:.2f}, "
                f"stop={stop_level:.2f} (entrada={entry_price:.2f}, ATR={atr:.2f})"
            )
        return triggered

    # ── Ratio b ───────────────────────────────────────────────────────────────

    def compute_gain_loss_ratio(
        self,
        returns: pd.Series,
        window: int = 60,
    ) -> float:
        """
        Calcula el ratio b = ganancia media / pérdida media sobre una ventana histórica.

        Este es el parámetro b de la fórmula de Kelly. Se recalcula en cada
        ventana de entrenamiento del walk-forward con los retornos históricos.

        NOTA: Para activos con perfiles de retorno muy distintos (ej: bonos vs acciones
        tech), calcular b por separado para cada clase de activo con los retornos
        históricos de esa clase específica.

        Args:
            returns: Serie de retornos diarios (de un ticker o clase)
            window: Número de días a usar para el cálculo

        Returns:
            b: Ratio > 0. Si no hay suficientes datos, retorna 1.0 (neutral).
        """
        recent = returns.iloc[-window:] if len(returns) >= window else returns
        gains = recent[recent > 0]
        losses = recent[recent < 0].abs()

        if gains.empty or losses.empty:
            return 1.0

        return float(gains.mean() / losses.mean())
