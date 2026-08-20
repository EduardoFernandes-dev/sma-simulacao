"""
Agente Aleatório - Baseline para comparação.
"""

import random
from typing import TYPE_CHECKING

from src.core.agente import Agente
from src.core.accao import Accao, TipoAccao, ACCOES_MOVIMENTO

if TYPE_CHECKING:
    from src.core.observacao import Observacao


class AgenteAleatorio(Agente):
    """
    Agente que escolhe ações aleatoriamente.
    
    Serve como baseline para comparação com agentes com aprendizagem.
    """
    
    def __init__(self, nome: str = "Aleatorio", posicao: tuple = (0, 0)):
        super().__init__(nome, posicao)
    
    def age(self) -> Accao:
        """Escolhe uma ação aleatória."""
        return random.choice(ACCOES_MOVIMENTO)
    
    def __str__(self) -> str:
        return f"AgenteAleatorio({self.nome})"
