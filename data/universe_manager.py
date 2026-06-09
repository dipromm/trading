"""
Gestor del universo de activos en producción.

Aplica los cambios aprobados por el humano al universo activo,
manteniendo un log auditable de cada decisión con fecha y razón.

SEPARACIÓN CRÍTICA entre los dos universos del proyecto:

    data/universe_2018-01-01.csv  → Universo del BACKTEST
                                    Fijado a fecha de inicio del proyecto.
                                    NUNCA se modifica. Garantiza resultados reproducibles.

    data/universe_live.csv        → Universo de PAPER TRADING / PRODUCCIÓN
                                    Evoluciona con las aprobaciones del Explorador.
                                    Si no existe, se inicializa como copia del backtest.

    logs/universe_changes.jsonl   → Log auditable de todos los cambios aprobados.
                                    Formato: {fecha, accion, ticker, asset_class,
                                              razon, aprobado_por, recomendacion_id}

Uso:
    manager = UniverseManager()
    manager.apply_change(action="ADD", ticker="USO", asset_class="commodity",
                         reason="correlacion_negativa_cartera=-0.31")
    manager.apply_change(action="REMOVE", ticker="CHKP",
                         reason="f*=0 el 94% de los dias de paper trading")
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from utils.config_loader import load_config

logger = logging.getLogger(__name__)

BACKTEST_UNIVERSE = Path("data/universe_2018-01-01.csv")
LIVE_UNIVERSE = Path("data/universe_live.csv")
CHANGES_LOG = Path("logs/universe_changes.jsonl")


class UniverseManager:
    """
    Gestiona el universo activo de producción con log auditable.
    El universo del backtest histórico nunca se toca.
    """

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = load_config()
        self._ensure_live_universe_exists()

    def _ensure_live_universe_exists(self) -> None:
        """
        Si universe_live.csv no existe, lo crea como copia del universo de backtest.
        Esto ocurre la primera vez que se pasa de backtest a paper trading.
        """
        if not LIVE_UNIVERSE.exists():
            if not BACKTEST_UNIVERSE.exists():
                raise FileNotFoundError(
                    f"Universo de backtest no encontrado: {BACKTEST_UNIVERSE}"
                )
            shutil.copy(BACKTEST_UNIVERSE, LIVE_UNIVERSE)
            logger.info(
                f"universe_live.csv inicializado como copia de {BACKTEST_UNIVERSE.name}"
            )

    def load_live_universe(self) -> pd.DataFrame:
        """Carga el universo activo de producción."""
        return pd.read_csv(LIVE_UNIVERSE)

    def load_backtest_universe(self) -> pd.DataFrame:
        """Carga el universo fijo de backtest (solo lectura)."""
        return pd.read_csv(BACKTEST_UNIVERSE)

    def apply_change(
        self,
        action: Literal["ADD", "REMOVE"],
        ticker: str,
        asset_class: str = "equity",
        reason: str = "",
        approved_by: str = "human",
        name: str = "",
        sector: str = "",
        notes: str = "",
    ) -> None:
        """
        Aplica un cambio aprobado al universo activo.

        Args:
            action: "ADD" para añadir un nuevo activo, "REMOVE" para retirar uno
            ticker: Símbolo del activo (ej: "USO", "CHKP")
            asset_class: Clase del activo (solo para ADD)
            reason: Razón documentada del cambio (de El Explorador)
            approved_by: Quién aprobó el cambio ("human" siempre)
            name: Nombre del activo (solo para ADD)
            sector: Sector del activo (solo para ADD)
            notes: Notas adicionales (solo para ADD)
        """
        df = self.load_live_universe()

        if action == "ADD":
            if ticker in df["ticker"].values:
                logger.warning(f"[{ticker}] Ya está en el universo activo. No se añade.")
                return
            new_row = pd.DataFrame([{
                "ticker": ticker,
                "name": name or ticker,
                "asset_class": asset_class,
                "sector": sector,
                "notes": notes or f"Añadido el {datetime.now().strftime('%Y-%m-%d')}. {reason}",
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            logger.info(f"[ADD] {ticker} ({asset_class}) añadido al universo activo")

        elif action == "REMOVE":
            if ticker not in df["ticker"].values:
                logger.warning(f"[{ticker}] No está en el universo activo. No se puede retirar.")
                return
            df = df[df["ticker"] != ticker]
            logger.info(f"[REMOVE] {ticker} retirado del universo activo")

        df.to_csv(LIVE_UNIVERSE, index=False)
        self._log_change(action, ticker, asset_class, reason, approved_by)

    def _log_change(
        self,
        action: str,
        ticker: str,
        asset_class: str,
        reason: str,
        approved_by: str,
    ) -> None:
        """Registra el cambio en el log auditable."""
        CHANGES_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accion": action,
            "ticker": ticker,
            "asset_class": asset_class,
            "razon": reason,
            "aprobado_por": approved_by,
        }
        with open(CHANGES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def diff_universes(self) -> dict:
        """
        Compara el universo de backtest con el universo activo.

        Returns:
            dict con {added: [...], removed: [...]} respecto al backtest original.
        """
        backtest = set(self.load_backtest_universe()["ticker"].tolist())
        live = set(self.load_live_universe()["ticker"].tolist())
        return {
            "added_since_backtest": sorted(live - backtest),
            "removed_since_backtest": sorted(backtest - live),
            "total_live": len(live),
            "total_backtest": len(backtest),
        }

    def get_change_history(self) -> list[dict]:
        """Carga el historial completo de cambios del universo."""
        if not CHANGES_LOG.exists():
            return []
        history = []
        with open(CHANGES_LOG, "r", encoding="utf-8") as f:
            for line in f:
                history.append(json.loads(line.strip()))
        return history
