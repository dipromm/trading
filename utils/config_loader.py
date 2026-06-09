"""
Cargador centralizado de configuración.

Todas las partes del proyecto deben leer parámetros a través de load_config(),
nunca hardcodeando valores directamente en el código.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_cached_config: Optional[dict] = None


def load_config(path: Optional[str] = None, force_reload: bool = False) -> dict:
    """
    Carga y cachea el config.yaml del proyecto.

    Args:
        path: Ruta alternativa al config (útil en tests).
        force_reload: Si True, ignora el caché y recarga desde disco.

    Returns:
        Diccionario con toda la configuración del proyecto.
    """
    global _cached_config

    if _cached_config is not None and not force_reload and path is None:
        return _cached_config

    config_path = Path(path) if path else _CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml no encontrado en {config_path}. "
            "Asegúrate de ejecutar los scripts desde la raíz del proyecto."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if path is None:
        _cached_config = config

    logger.debug(f"Configuración cargada desde {config_path}")
    return config
