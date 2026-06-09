"""
Reproducibilidad global del proyecto.

Llamar a set_all_seeds() al inicio de cada script/notebook garantiza
que cualquier persona que clone el repositorio obtenga exactamente los
mismos resultados con el mismo config.yaml.
"""

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """
    Fija el seed en todo el stack: Python, NumPy y PyTorch (CPU + CUDA).

    Args:
        seed: Valor del seed. Leer siempre de config['general']['random_seed'].
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("PyTorch no disponible; seeds de CUDA omitidas.")

    logger.debug(f"Seeds fijadas a {seed} en todo el stack.")
