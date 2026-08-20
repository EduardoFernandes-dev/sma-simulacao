"""
Agente SARSA - Aprendizagem por reforço on-policy.
"""

import random
import pickle
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict
from pathlib import Path

from src.core.agente import Agente
from src.core.accao import Accao, ACCOES_MOVIMENTO

if TYPE_CHECKING:
    from src.core.observacao import Observacao


class AgenteSARSA(Agente):    
    def __init__(self, nome: str = "SARSA", posicao: tuple = (0, 0),
                 alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        super().__init__(nome, posicao)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_inicial = epsilon
        
        # Tabela Q: {estado: {ação: valor}}
        self.q_table: Dict[Tuple, Dict[Accao, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACCOES_MOVIMENTO}
        )
        
        # Estado e ação anteriores para update SARSA
        self._estado_anterior: Optional[Tuple] = None
        self._accao_anterior: Optional[Accao] = None
        
        # Próxima ação (característica do SARSA)
        self._proxima_accao: Optional[Accao] = None
    
    def age(self) -> Accao:
        if self._observacao_atual is None:
            return random.choice(ACCOES_MOVIMENTO)
        
        estado = self._observacao_atual.para_estado()
        
        # Se já temos a próxima ação definida (do update SARSA), usá-la
        if self._proxima_accao is not None:
            accao = self._proxima_accao
            self._proxima_accao = None
        else:
            # Escolher ação usando ε-greedy
            accao = self._escolher_accao(estado)
        
        # Guardar para update
        self._estado_anterior = estado
        self._accao_anterior = accao
        
        return accao
    
    def _escolher_accao(self, estado: Tuple) -> Accao:
        """Escolhe ação usando política ε-greedy."""
        if self.modo_aprendizagem and random.random() < self.epsilon:
            # Exploração: ação aleatória
            return random.choice(ACCOES_MOVIMENTO)
        else:
            # Exploitation: melhor ação conhecida
            return self._melhor_accao(estado)
    
    def _melhor_accao(self, estado: Tuple) -> Accao:
        """Retorna a ação com maior valor Q para o estado."""
        valores = self.q_table[estado]
        max_valor = max(valores.values())
        
        # Se houver empate, escolher aleatoriamente
        melhores = [a for a, v in valores.items() if v == max_valor]
        return random.choice(melhores)
    
    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        super().avaliacaoEstadoAtual(recompensa)
        
        if not self.modo_aprendizagem:
            return
        
        if self._estado_anterior is None or self._accao_anterior is None:
            return
        
        # Novo estado
        estado_novo = self._observacao_atual.para_estado() if self._observacao_atual else None
        
        if estado_novo:
            # Escolher a próxima ação AGORA (característica do SARSA)
            self._proxima_accao = self._escolher_accao(estado_novo)
            q_proximo = self.q_table[estado_novo][self._proxima_accao]
        else:
            q_proximo = 0
        
        # Update SARSA: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
        q_atual = self.q_table[self._estado_anterior][self._accao_anterior]
        q_novo = q_atual + self.alpha * (recompensa + self.gamma * q_proximo - q_atual)
        self.q_table[self._estado_anterior][self._accao_anterior] = q_novo
    
    def fim_episodio(self) -> None:
        """Chamado no fim de cada episódio."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reset(self, posicao: tuple = None) -> None:
        """Reinicia o agente para novo episódio."""
        super().reset(posicao)
        self._estado_anterior = None
        self._accao_anterior = None
        self._proxima_accao = None
        self.fim_episodio()
    
    def guardar_modelo(self, caminho: str) -> None:
        """Guarda a tabela Q num ficheiro."""
        caminho = Path(caminho)
        with open(caminho, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Modelo SARSA guardado em: {caminho}")
    
    def carregar_modelo(self, caminho: str) -> None:
        """Carrega a tabela Q de um ficheiro."""
        caminho = Path(caminho)
        with open(caminho, 'rb') as f:
            tabela = pickle.load(f)
        self.q_table = defaultdict(
            lambda: {a: 0.0 for a in ACCOES_MOVIMENTO},
            tabela
        )
        print(f"Modelo SARSA carregado de: {caminho}")
    
    def estatisticas(self) -> Dict:
        """Retorna estatísticas do agente."""
        return {
            'estados_conhecidos': len(self.q_table),
            'epsilon_atual': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
    
    def __str__(self) -> str:
        return f"AgenteSARSA({self.nome}, ε={self.epsilon:.3f})"
