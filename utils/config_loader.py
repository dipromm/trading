"""
Cargador centralizado de configuración.

Todas las partes del proyecto deben leer parámetros a través de load_config(),
nunca hardcodeando valores directamente en el código.
"""

import copy
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_cached_config: Optional[dict] = None


def merge_config(base: dict, override: dict) -> dict:
    """Deep-merge de override sobre base (override gana en conflictos)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: Optional[str] = None,
    profile_path: Optional[str] = None,
    force_reload: bool = False,
) -> dict:
    """
    Carga y cachea el config.yaml del proyecto.

    Args:
        path: Ruta alternativa al config (útil en tests).
        profile_path: YAML de perfil que sobreescribe claves del config base.
        force_reload: Si True, ignora el caché y recarga desde disco.

    Returns:
        Diccionario con toda la configuración del proyecto.
    """
    global _cached_config

    use_cache = (
        _cached_config is not None
        and not force_reload
        and path is None
        and profile_path is None
    )
    if use_cache:
        return _cached_config

    config_path = Path(path) if path else _CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml no encontrado en {config_path}. "
            "Asegúrate de ejecutar los scripts desde la raíz del proyecto."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if profile_path is not None:
        profile_file = Path(profile_path)
        if not profile_file.exists():
            raise FileNotFoundError(f"Perfil no encontrado: {profile_file}")
        with open(profile_file, "r", encoding="utf-8") as f:
            profile_cfg = yaml.safe_load(f) or {}
        config = merge_config(config, profile_cfg)
        logger.info("Perfil aplicado: %s", profile_file)

    # Cachear siempre que se use el config.yaml base (sin ruta alternativa).
    # Cuando se aplica un perfil, el cache se actualiza con el config merged
    # para que módulos que llaman load_config() internamente (BacktestEngine,
    # GestorRiesgos, etc.) reciban los overrides del perfil activo.
    if path is None:
        _cached_config = config

    logger.debug("Configuración cargada desde %s", config_path)
    return config
