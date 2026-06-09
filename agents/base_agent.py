"""
Interfaz común para todos los agentes del Consejo.

Todos los agentes deben heredar de AgentBase e implementar fit() y predict().
Esto garantiza que el Juez pueda tratarlos de forma uniforme.
"""

from abc import ABC, abstractmethod

import pandas as pd


class AgentBase(ABC):
    """Clase base abstracta para los agentes del Consejo."""

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entrena el agente con datos históricos.

        Args:
            train_data: DataFrame con features y columna 'target_binary'.
                        Solo contiene datos del período de entrenamiento
                        de la ventana walk-forward actual.
        """
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera la señal del agente para el período dado.

        Args:
            data: DataFrame con las mismas features usadas en fit().

        Returns:
            Serie con la señal del agente por fecha.
            - Agentes predictores (Matemático, Analista): probabilidad calibrada ∈ [0, 1]
            - El Conspiranoico: 1 = anomalía detectada (veto), 0 = mercado normal
            - El Cazador: 1 = alerta de insider, 0 = sin alerta
        """
        ...

    def is_fitted(self) -> bool:
        """Devuelve True si el agente ya ha sido entrenado."""
        return getattr(self, "_is_fitted", False)
