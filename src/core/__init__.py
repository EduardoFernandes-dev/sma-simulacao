"""
Módulo Core do Simulador SMA.
Contém as classes base: Simulador, Ambiente, Agente.
"""

from .simulador import MotorDeSimulacao
from .ambiente import Ambiente
from .agente import Agente
from .observacao import Observacao
from .accao import Accao

__all__ = ['MotorDeSimulacao', 'Ambiente', 'Agente', 'Observacao', 'Accao']
