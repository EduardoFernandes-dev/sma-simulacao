"""
Classe Abstrata Ambiente - Interface base para todos os ambientes.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import threading
import json
import numpy as np

from .observacao import Observacao
from .accao import Accao
from .agente import Agente


class TipoCelula:
    """Tipos de células possíveis no ambiente."""
    LIVRE = 'livre'
    OBSTACULO = 'obstaculo'
    PAREDE = 'parede'
    OBJETIVO = 'objetivo'  # Farol ou Saída
    INICIO = 'inicio'


class Ambiente(ABC):
    def __init__(self, largura: int, altura: int):
        self.largura = largura
        self.altura = altura
        self.grid: np.ndarray = np.full((altura, largura), TipoCelula.LIVRE, dtype=object)
        self.agentes: List[Agente] = []
        self.objetivo: Optional[Tuple[int, int]] = None
        self.posicao_inicial: Optional[Tuple[int, int]] = None
        
        # Lock para acesso thread-safe ao grid
        self._lock = threading.Lock()
        
        # Métricas
        self.passos_episodio = 0
        self.episodio_terminado = False
    
    @classmethod
    def cria(cls, nome_ficheiro_parametros: str) -> 'Ambiente':
        with open(nome_ficheiro_parametros, 'r') as f:
            config = json.load(f)
        # Subclasses implementam a criação específica
        raise NotImplementedError("Subclasses devem implementar cria()")
    
    @abstractmethod
    def observacaoPara(self, agente: Agente) -> Observacao:
        pass
    
    def atualizacao(self) -> None:
        self.passos_episodio += 1
    
    def agir(self, accao: Accao, agente: Agente) -> Tuple[bool, float]:
        with self._lock:
            dx, dy = accao.delta
            nova_x = agente.posicao[0] + dx
            nova_y = agente.posicao[1] + dy
            
            # Verificar limites
            if not self._dentro_limites(nova_x, nova_y):
                return False, self._recompensa_colisao()
            
            # Verificar obstáculos
            if self._e_obstaculo(nova_x, nova_y):
                return False, self._recompensa_colisao()
            
            # Mover agente
            agente.posicao = (nova_x, nova_y)
            
            # Calcular recompensa
            recompensa = self._calcular_recompensa(agente)
            
            # Verificar se chegou ao objetivo
            if self._chegou_objetivo(agente):
                self.episodio_terminado = True
            
            return True, recompensa
    
    def adicionar_agente(self, agente: Agente, posicao: Tuple[int, int] = None) -> None:
        if posicao is None:
            posicao = self.posicao_inicial or (0, 0)
        agente.posicao = posicao
        self.agentes.append(agente)
    
    def reset(self) -> None:
        """Reinicia o ambiente para um novo episódio."""
        self.passos_episodio = 0
        self.episodio_terminado = False
        
        # Reposicionar agentes
        for agente in self.agentes:
            agente.reset(self.posicao_inicial)
    
    # Métodos auxiliares
    def _dentro_limites(self, x: int, y: int) -> bool:
        """Verifica se posição está dentro dos limites."""
        return 0 <= x < self.largura and 0 <= y < self.altura
    
    def _e_obstaculo(self, x: int, y: int) -> bool:
        """Verifica se célula é obstáculo ou parede."""
        tipo = self.grid[y, x]
        return tipo in [TipoCelula.OBSTACULO, TipoCelula.PAREDE]
    
    def _chegou_objetivo(self, agente: Agente) -> bool:
        """Verifica se agente chegou ao objetivo."""
        return self.objetivo and agente.posicao == self.objetivo
    
    def _obter_vizinhanca(self, x: int, y: int) -> Dict[str, str]:
        vizinhanca = {}
        direcoes = {
            'N': (0, -1),
            'S': (0, 1),
            'E': (1, 0),
            'O': (-1, 0)
        }
        
        for direcao, (dx, dy) in direcoes.items():
            nx, ny = x + dx, y + dy
            if not self._dentro_limites(nx, ny):
                vizinhanca[direcao] = TipoCelula.PAREDE
            else:
                vizinhanca[direcao] = self.grid[ny, nx]
        
        return vizinhanca
    
    def _calcular_direcao_objetivo(self, posicao: Tuple[int, int]) -> Optional[str]:
        if self.objetivo is None:
            return None
        
        dx = self.objetivo[0] - posicao[0]
        dy = self.objetivo[1] - posicao[1]
        
        if dx == 0 and dy == 0:
            return 'OBJETIVO'
        
        direcao = ''
        if dy < 0:
            direcao += 'N'
        elif dy > 0:
            direcao += 'S'
        
        if dx > 0:
            direcao += 'E'
        elif dx < 0:
            direcao += 'O'
        
        return direcao if direcao else None
    
    def _distancia_objetivo(self, posicao: Tuple[int, int]) -> float:
        """Calcula distância Manhattan ao objetivo."""
        if self.objetivo is None:
            return float('inf')
        return abs(posicao[0] - self.objetivo[0]) + abs(posicao[1] - self.objetivo[1])
    
    # Métodos de recompensa (subclasses podem override)
    @abstractmethod
    def _calcular_recompensa(self, agente: Agente) -> float:
        """Calcula recompensa baseada no estado atual."""
        pass
    
    def _recompensa_colisao(self) -> float:
        """Recompensa por colidir com obstáculo/parede."""
        return -5.0
    
    def __str__(self) -> str:
        return f"Ambiente({self.largura}x{self.altura}, agentes={len(self.agentes)})"
