"""
Classe Ação - Define as ações possíveis dos agentes.
"""

from enum import Enum
from typing import Tuple


class TipoAccao(Enum):
    """Tipos de ações disponíveis para os agentes."""
    NORTE = 0
    SUL = 1
    ESTE = 2
    OESTE = 3
    PARADO = 4
    
    @classmethod
    def lista_movimentos(cls) -> list:
        """Retorna apenas as ações de movimento (exclui PARADO)."""
        return [cls.NORTE, cls.SUL, cls.ESTE, cls.OESTE]
    
    def delta(self) -> Tuple[int, int]:
        deltas = {
            TipoAccao.NORTE: (0, -1),
            TipoAccao.SUL: (0, 1),
            TipoAccao.ESTE: (1, 0),
            TipoAccao.OESTE: (-1, 0),
            TipoAccao.PARADO: (0, 0)
        }
        return deltas[self]


class Accao:
    def __init__(self, tipo: TipoAccao, parametros: dict = None):
        self.tipo = tipo
        self.parametros = parametros or {}
    
    @property
    def delta(self) -> Tuple[int, int]:
        """Retorna o delta de movimento desta ação."""
        return self.tipo.delta()
    
    def __eq__(self, other):
        if isinstance(other, Accao):
            return self.tipo == other.tipo
        return False
    
    def __hash__(self):
        return hash(self.tipo)
    
    def __str__(self):
        return f"Accao({self.tipo.name})"
    
    def __repr__(self):
        return self.__str__()


NORTE = Accao(TipoAccao.NORTE)
SUL = Accao(TipoAccao.SUL)
ESTE = Accao(TipoAccao.ESTE)
OESTE = Accao(TipoAccao.OESTE)
PARADO = Accao(TipoAccao.PARADO)

ACCOES_MOVIMENTO = [NORTE, SUL, ESTE, OESTE]
TODAS_ACCOES = [NORTE, SUL, ESTE, OESTE, PARADO]
