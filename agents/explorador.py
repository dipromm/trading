"""
El Explorador — Gestión dinámica del universo de activos.

Opera SOLO en paper trading / producción.
NUNCA modifica el universo del backtest histórico (fijado en data/universe_2018-01-01.csv).

Responsabilidades:
    1. Analizar el historial de paper trades para detectar activos candidatos a RETIRAR
    2. Evaluar candidatos externos para detectar oportunidades de AÑADIR al universo
    3. Generar recomendaciones para revisión humana (Human-in-the-Loop obligatorio)

Separación crítica:
    BACKTEST  → universo fijo en data/universe_2018-01-01.csv — NO TOCAR NUNCA
    PRODUCCIÓN → universo activo en data/universe_live.csv — evoluciona con aprobaciones

Fuentes de información:
    Para RETIRAR: logs/trades/ (historial de paper trades del sistema)
    Para AÑADIR:  yfinance (evalúa candidatos externos con datos de mercado)
    Para AÑADIR NO usa logs de trades — si nunca has operado un activo,
    tus trades no contienen información sobre él.

Implementar en Fase 9.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Umbrales configurables en config.yaml → explorador
_NEUTRAL_SIGNAL_DAYS_THRESHOLD = 0.90   # Si f*=0 más del 90% de días → candidato a retirar
_MIN_PREDICTION_ACCURACY = 0.50         # Si accuracy < 50% → peor que azar → candidato a retirar
_MAX_CORRELATION_WITH_EXISTING = 0.85   # Si correlación > 85% con activo ya presente → redundante
_MIN_CORRELATION_BENEFIT = -0.10        # Candidato externo útil si correlación < -0.10 con cartera


class Explorador:
    """
    Monitorea el historial de paper trades y propone cambios al universo activo.
    Las propuestas requieren aprobación humana antes de aplicarse.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config.get("explorador", {})
        self.trades_dir = Path(config["general"]["log_dir"]) / "trades"
        self.universe_live = Path("data/universe_live.csv")
        self.neutral_threshold = self.config.get(
            "neutral_signal_days_threshold", _NEUTRAL_SIGNAL_DAYS_THRESHOLD
        )
        self.min_accuracy = self.config.get(
            "min_prediction_accuracy", _MIN_PREDICTION_ACCURACY
        )

    # ── Análisis de trades existentes (candidatos a RETIRAR) ──────────────────

    def load_trade_history(self) -> pd.DataFrame:
        """
        Carga el historial de paper trades desde logs/trades/*.jsonl.

        Returns:
            DataFrame con todas las operaciones registradas.
        """
        # TODO: implementar en Fase 9
        # Leer todos los .jsonl de self.trades_dir
        # Parsear cada línea como JSON (TradeLog del BacktestEngine)
        # Retornar DataFrame con columnas: date, ticker, action, kelly_fraction,
        #   agent_votes (expandido), result (calculado post-hoc)
        raise NotImplementedError("Implementar en Fase 9")

    def find_removal_candidates(self, trade_history: pd.DataFrame) -> list[dict]:
        """
        Analiza el historial de trades y detecta activos que no aportan valor.

        Criterios para proponer RETIRAR un activo:
            1. f* ≈ 0 más del NEUTRAL_THRESHOLD% de los días → sin señal útil
            2. Accuracy del Matemático < MIN_ACCURACY en ese ticker → peor que azar
            3. Correlación > MAX_CORRELATION con otro activo del universo → redundante

        Returns:
            Lista de dicts con {ticker, razon, evidencia, urgencia}
        """
        # TODO: implementar en Fase 9
        # Para cada ticker en trade_history:
        #   1. Calcular % de días con kelly_fraction == 0
        #   2. Calcular accuracy de la predicción (comparar señal vs retorno real)
        #   3. Si supera umbrales → añadir a candidatos con razón y evidencia
        raise NotImplementedError("Implementar en Fase 9")

    # ── Evaluación de candidatos externos (candidatos a AÑADIR) ──────────────

    def evaluate_candidate(
        self,
        ticker: str,
        asset_class: str,
        current_prices: dict[str, pd.DataFrame],
    ) -> dict:
        """
        Evalúa si un activo externo aporta valor al universo actual.

        Criterios para proponer AÑADIR un activo:
            1. Correlación baja o negativa con la cartera actual (diversificación)
            2. Suficientes datos históricos disponibles (≥ 5 años en yfinance)
            3. Liquidez mínima (volumen medio diario > umbral)
            4. No es redundante con activos ya presentes (correlación < MAX_CORRELATION)

        Args:
            ticker: Símbolo del candidato (ej: "USO", "GDX")
            asset_class: Clase de activo del candidato
            current_prices: Precios actuales de la cartera para calcular correlaciones

        Returns:
            dict con {ticker, recomendacion, correlacion_media, datos_disponibles,
                     volumen_medio, razon}
        """
        # TODO: implementar en Fase 9
        # 1. Descargar datos del candidato via yfinance
        # 2. Calcular correlación de retornos con cada activo de la cartera
        # 3. Calcular correlación media con la cartera completa
        # 4. Verificar calidad y cantidad de datos
        # 5. Retornar evaluación con recomendación y evidencia
        raise NotImplementedError("Implementar en Fase 9")

    # ── Generación de recomendaciones ─────────────────────────────────────────

    def generate_recommendations(
        self,
        trade_history: pd.DataFrame,
        current_prices: dict[str, pd.DataFrame],
        external_candidates: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Genera la lista completa de recomendaciones para revisión humana.

        Returns:
            Lista de dicts con formato:
            {
                "accion": "RETIRAR" | "AÑADIR",
                "ticker": str,
                "asset_class": str,
                "razon": str,
                "evidencia": dict,   # métricas que soportan la recomendación
                "urgencia": "alta" | "media" | "baja"
            }
        """
        # TODO: implementar en Fase 9
        # 1. removal_candidates = self.find_removal_candidates(trade_history)
        # 2. Si external_candidates: evaluar cada uno con evaluate_candidate()
        # 3. Combinar y ordenar por urgencia
        raise NotImplementedError("Implementar en Fase 9")

    def save_recommendations(self, recommendations: list[dict]) -> Path:
        """Guarda las recomendaciones en logs/explorador_recommendations.jsonl."""
        output = Path(self.config.get("log_dir", "logs")) / "explorador_recommendations.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for rec in recommendations:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Recomendaciones guardadas: {output} ({len(recommendations)} items)")
        return output
