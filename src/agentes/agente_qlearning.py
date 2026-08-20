"""
Agente Q-Learning - Aprendizagem por reforço.
"""

import random
import json
import pickle
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict
from pathlib import Path

from src.core.agente import Agente
from src.core.accao import Accao, TipoAccao, ACCOES_MOVIMENTO

if TYPE_CHECKING:
    from src.core.observacao import Observacao


class AgenteQLearning(Agente):
    def __init__(self, nome: str = "QLearning", posicao: tuple = (0, 0),
                 alpha: float = 0.1, gamma: float = 0.95, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 alpha_decay: float = 1.0, alpha_min: float = 0.01):
        super().__init__(nome, posicao)
        
        # Hiperparâmetros
        self.alpha = alpha          # Taxa de aprendizagem
        self.alpha_inicial = alpha  # Para reset
        self.alpha_decay = alpha_decay  # Decaimento do alpha (1.0 = sem decay)
        self.alpha_min = alpha_min  # Valor mínimo do alpha
        
        self.gamma = gamma          # Fator de desconto
        self.epsilon = epsilon      # Exploração inicial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_inicial = epsilon
        
        # Tabela Q: {estado: {ação: valor}}
        self.q_table: Dict[Tuple, Dict[Accao, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACCOES_MOVIMENTO}
        )
        
        # Estado anterior para update
        self._estado_anterior: Optional[Tuple] = None
        self._accao_anterior: Optional[Accao] = None
    
    def age(self) -> Accao:
        if self._observacao_atual is None:
            return random.choice(ACCOES_MOVIMENTO)
        
        estado = self._observacao_atual.para_estado()
        
        # Política ε-greedy
        if self.modo_aprendizagem and random.random() < self.epsilon:
            # Exploração: ação aleatória
            accao = random.choice(ACCOES_MOVIMENTO)
        else:
            # Exploração: melhor ação conhecida
            accao = self._melhor_accao(estado)
        
        # Guardar para update
        self._estado_anterior = estado
        self._accao_anterior = accao
        
        return accao
    
    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        super().avaliacaoEstadoAtual(recompensa)
        
        if not self.modo_aprendizagem:
            return
        
        if self._estado_anterior is None or self._accao_anterior is None:
            return
        
        # Novo estado
        estado_novo = self._observacao_atual.para_estado() if self._observacao_atual else None
        
        # Update Q-Learning
        q_atual = self.q_table[self._estado_anterior][self._accao_anterior]
        
        if estado_novo:
            q_max_proximo = max(self.q_table[estado_novo].values())
        else:
            q_max_proximo = 0
        
        # Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        q_novo = q_atual + self.alpha * (recompensa + self.gamma * q_max_proximo - q_atual)
        self.q_table[self._estado_anterior][self._accao_anterior] = q_novo
    
    def _melhor_accao(self, estado: Tuple) -> Accao:
        valores = self.q_table[estado]
        max_valor = max(valores.values())
        
        # Se houver empate, escolher aleatoriamente entre as melhores
        melhores = [a for a, v in valores.items() if v == max_valor]
        return random.choice(melhores)
    
    def fim_episodio(self) -> None:
        # Decaimento do epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Decaimento do alpha (se ativo)
        if self.alpha_decay < 1.0 and self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay
            self.alpha = max(self.alpha, self.alpha_min)
    
    def reset(self, posicao: tuple = None) -> None:
        super().reset(posicao)
        self._estado_anterior = None
        self._accao_anterior = None
        self.fim_episodio()
    
    def guardar_modelo(self, caminho: str, seed: int = None) -> None:
        caminho = Path(caminho)
        dados = {
            'q_table': dict(self.q_table),
            'seed': seed,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        with open(caminho, 'wb') as f:
            pickle.dump(dados, f)
        print(f"Modelo guardado em: {caminho}" + (f" (seed={seed})" if seed else ""))
    
    def carregar_modelo(self, caminho: str) -> dict:
        caminho = Path(caminho)
        with open(caminho, 'rb') as f:
            dados = pickle.load(f)
        
        # Suportar formato antigo (apenas q_table) e novo (dict com seed)
        if isinstance(dados, dict) and 'q_table' in dados:
            tabela = dados['q_table']
            seed = dados.get('seed', None)
        else:
            tabela = dados
            seed = None
        
        self.q_table = defaultdict(
            lambda: {a: 0.0 for a in ACCOES_MOVIMENTO},
            tabela
        )
        print(f"Modelo carregado de: {caminho}" + (f" (seed={seed})" if seed else ""))
        return {'seed': seed}
    
    def estatisticas(self) -> Dict:
        return {
            'estados_conhecidos': len(self.q_table),
            'epsilon_atual': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
    
    def __str__(self) -> str:
        return f"AgenteQLearning({self.nome}, ε={self.epsilon:.3f}, α={self.alpha:.3f})"

