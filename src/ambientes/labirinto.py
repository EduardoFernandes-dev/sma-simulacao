"""
Ambiente Labirinto - Grelha 2D com paredes fixas geradas algoritmicamente.
"""

import json
import random
from typing import Tuple, Optional, List
from pathlib import Path

from src.core.ambiente import Ambiente, TipoCelula
from src.core.observacao import Observacao
from src.core.agente import Agente


class AmbienteLabirinto(Ambiente):
    """
    Ambiente do problema do Labirinto.
    
    Uma grelha 2D com paredes fixas geradas algoritmicamente (DFS),
    onde o agente deve navegar da entrada até à saída.
    
    Attributes:
        recompensa_objetivo: Recompensa por chegar à saída
        penalidade_passo: Penalidade por cada passo
        penalidade_colisao: Penalidade por colidir com parede
    """
    
    def __init__(self, largura: int = 21, altura: int = 21):
        """
        Inicializa o labirinto.
        
        Args:
            largura: Largura da grelha (deve ser ímpar para DFS)
            altura: Altura da grelha (deve ser ímpar para DFS)
        """
        # Garantir dimensões ímpares para o algoritmo DFS
        largura = largura if largura % 2 == 1 else largura + 1
        altura = altura if altura % 2 == 1 else altura + 1
        
        super().__init__(largura, altura)
        
        # Parâmetros de recompensa
        self.recompensa_objetivo = 100.0
        self.penalidade_passo = -0.1
        self.penalidade_colisao = -5.0
        self.recompensa_aproximar = 1.0
        self.penalidade_afastar = -1.0
        
        # Tracking de distância
        self._distancia_anterior = {}
    
    @classmethod
    def cria(cls, nome_ficheiro_parametros: str) -> 'AmbienteLabirinto':
        """Cria ambiente a partir de ficheiro de configuração."""
        caminho = Path(nome_ficheiro_parametros)
        with open(caminho, 'r') as f:
            config = json.load(f)
        
        ambiente = cls(
            largura=config.get('largura', 21),
            altura=config.get('altura', 21)
        )
        
        # Posições
        if 'inicio' in config:
            ambiente.posicao_inicial = tuple(config['inicio'])
        else:
            ambiente.posicao_inicial = (1, 1)
        
        if 'saida' in config:
            ambiente.objetivo = tuple(config['saida'])
        else:
            ambiente.objetivo = (ambiente.largura - 2, ambiente.altura - 2)
        
        # Recompensas customizadas
        if 'recompensas' in config:
            r = config['recompensas']
            ambiente.recompensa_objetivo = r.get('objetivo', 100.0)
            ambiente.penalidade_passo = r.get('passo', -0.1)
            ambiente.penalidade_colisao = r.get('colisao', -5.0)
        
        # Gerar labirinto
        if 'layout' in config:
            ambiente._carregar_layout(config['layout'])
        else:
            ambiente._gerar_labirinto_dfs()
        
        return ambiente
    
    def _gerar_labirinto_dfs(self, abertura: float = 0.3) -> None:
        """
        Gera um labirinto usando o algoritmo DFS (Depth-First Search).
        
        Args:
            abertura: Percentagem de paredes internas a remover (0.0-1.0).
                     0.0 = labirinto clássico, 1.0 = quase sem paredes.
                     Default 0.3 para mais espaço de exploração.
        """
        import numpy as np
        
        # Inicializar todas as células como parede
        self.grid = np.full((self.altura, self.largura), TipoCelula.PAREDE, dtype=object)
        
        # Stack para DFS
        stack: List[Tuple[int, int]] = []
        
        # Começar em (1, 1)
        start_x, start_y = 1, 1
        self.grid[start_y, start_x] = TipoCelula.LIVRE
        stack.append((start_x, start_y))
        
        # Direções: (dx, dy) - movemos 2 células de cada vez
        direcoes = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        
        while stack:
            x, y = stack[-1]
            
            # Encontrar vizinhos não visitados
            vizinhos = []
            for dx, dy in direcoes:
                nx, ny = x + dx, y + dy
                if 1 <= nx < self.largura - 1 and 1 <= ny < self.altura - 1:
                    if self.grid[ny, nx] == TipoCelula.PAREDE:
                        vizinhos.append((nx, ny, dx // 2, dy // 2))
            
            if vizinhos:
                # Escolher vizinho aleatório
                nx, ny, wx, wy = random.choice(vizinhos)
                
                # Remover parede entre células
                self.grid[y + wy, x + wx] = TipoCelula.LIVRE
                self.grid[ny, nx] = TipoCelula.LIVRE
                
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()
        
        # Remover paredes adicionais para abrir o labirinto
        if abertura > 0:
            self._abrir_labirinto(abertura)
        
        # Definir posição inicial e objetivo
        if self.posicao_inicial is None:
            self.posicao_inicial = (1, 1)
        if self.objetivo is None:
            self.objetivo = (self.largura - 2, self.altura - 2)
        
        # Marcar na grelha
        self.grid[self.posicao_inicial[1], self.posicao_inicial[0]] = TipoCelula.INICIO
        self.grid[self.objetivo[1], self.objetivo[0]] = TipoCelula.OBJETIVO
    
    def _abrir_labirinto(self, abertura: float) -> None:
        """Remove paredes internas aleatoriamente para criar mais espaço."""
        paredes_internas = []
        
        # Encontrar todas as paredes internas (não nas bordas)
        for y in range(1, self.altura - 1):
            for x in range(1, self.largura - 1):
                if self.grid[y, x] == TipoCelula.PAREDE:
                    paredes_internas.append((x, y))
        
        # Remover uma percentagem das paredes
        num_remover = int(len(paredes_internas) * abertura)
        paredes_a_remover = random.sample(paredes_internas, min(num_remover, len(paredes_internas)))
        
        for x, y in paredes_a_remover:
            self.grid[y, x] = TipoCelula.LIVRE
    
    def _carregar_layout(self, layout: List[str]) -> None:
        """
        Carrega layout de labirinto a partir de strings.
        
        Formato:
            '#' = parede
            '.' = livre
            'S' = início
            'E' = saída/objetivo
        """
        import numpy as np
        
        for y, linha in enumerate(layout):
            for x, char in enumerate(linha):
                if x >= self.largura or y >= self.altura:
                    continue
                
                if char == '#':
                    self.grid[y, x] = TipoCelula.PAREDE
                elif char == '.':
                    self.grid[y, x] = TipoCelula.LIVRE
                elif char == 'S':
                    self.grid[y, x] = TipoCelula.INICIO
                    self.posicao_inicial = (x, y)
                elif char == 'E':
                    self.grid[y, x] = TipoCelula.OBJETIVO
                    self.objetivo = (x, y)
    
    def observacaoPara(self, agente: Agente) -> Observacao:
        """Gera observação para o agente."""
        x, y = agente.posicao
        
        return Observacao(
            posicao=(x, y),
            vizinhanca=self._obter_vizinhanca(x, y),
            direcao_objetivo=self._calcular_direcao_objetivo((x, y)),
            distancia_objetivo=self._distancia_objetivo((x, y))
        )
    
    def _calcular_recompensa(self, agente: Agente) -> float:
        """Calcula recompensa baseada no estado atual."""
        # Chegou ao objetivo
        if self._chegou_objetivo(agente):
            return self.recompensa_objetivo
        
        # Recompensa de aproximação/afastamento
        distancia_atual = self._distancia_objetivo(agente.posicao)
        agente_id = id(agente)
        
        recompensa = self.penalidade_passo
        
        if agente_id in self._distancia_anterior:
            distancia_anterior = self._distancia_anterior[agente_id]
            if distancia_atual < distancia_anterior:
                recompensa += self.recompensa_aproximar
            elif distancia_atual > distancia_anterior:
                recompensa += self.penalidade_afastar
        
        self._distancia_anterior[agente_id] = distancia_atual
        
        return recompensa
    
    def _recompensa_colisao(self) -> float:
        """Recompensa por colidir."""
        return self.penalidade_colisao
    
    def reset(self) -> None:
        """Reinicia o ambiente."""
        super().reset()
        self._distancia_anterior.clear()
    
    def visualizar_texto(self) -> str:
        """Retorna representação textual do labirinto."""
        simbolos = {
            TipoCelula.LIVRE: ' ',
            TipoCelula.PAREDE: '█',
            TipoCelula.OBJETIVO: 'E',
            TipoCelula.INICIO: 'S'
        }
        
        linhas = []
        for y in range(self.altura):
            linha = ''
            for x in range(self.largura):
                agente_aqui = any(a.posicao == (x, y) for a in self.agentes)
                if agente_aqui:
                    linha += 'A'
                else:
                    linha += simbolos.get(self.grid[y, x], '?')
            linhas.append(linha)
        
        return '\n'.join(linhas)
