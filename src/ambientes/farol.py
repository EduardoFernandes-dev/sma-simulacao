"""
Ambiente Farol - Grelha 2D com farol e obstáculos.
"""

import json
import math
import random
from typing import Tuple, Optional
from pathlib import Path

from src.core.ambiente import Ambiente, TipoCelula
from src.core.observacao import Observacao
from src.core.agente import Agente


class AmbienteFarol(Ambiente):
    def __init__(self, largura: int = 10, altura: int = 10, 
                 densidade_obstaculos: float = 0.1):
        super().__init__(largura, altura)
        
        self.densidade_obstaculos = densidade_obstaculos
        
        # Parâmetros de recompensa
        self.recompensa_objetivo = 100.0
        self.penalidade_passo = -0.1
        self.penalidade_colisao = -5.0
        self.recompensa_aproximar = 1.0
        self.penalidade_afastar = -1.0
        
        # Tracking de distância para recompensa de aproximação
        self._distancia_anterior = {}
        
        # Sistema de feixe de luz rotativo
        self.angulo_feixe = 0.0          # Ângulo atual em graus (0-360)
        self.velocidade_rotacao = 15.0   # Graus por passo
        self.raio_feixe = 12              # Alcance em células
        self.abertura_feixe = 30.0       # Largura do cone em graus
        self.feixe_ativo = True          # Liga/desliga o feixe
    
    @classmethod
    def cria(cls, nome_ficheiro_parametros: str) -> 'AmbienteFarol':
        """Cria ambiente a partir de ficheiro de configuração."""
        caminho = Path(nome_ficheiro_parametros)
        with open(caminho, 'r') as f:
            config = json.load(f)
        
        ambiente = cls(
            largura=config.get('largura', 10),
            altura=config.get('altura', 10),
            densidade_obstaculos=config.get('densidade_obstaculos', 0.1)
        )
        
        # Posição do farol
        if 'farol' in config:
            ambiente.objetivo = tuple(config['farol'])
        else:
            # Farol no canto oposto por defeito
            ambiente.objetivo = (ambiente.largura - 1, ambiente.altura - 1)
        
        # Posição inicial
        if 'inicio' in config:
            ambiente.posicao_inicial = tuple(config['inicio'])
        else:
            ambiente.posicao_inicial = (0, 0)
        
        # Recompensas customizadas
        if 'recompensas' in config:
            r = config['recompensas']
            ambiente.recompensa_objetivo = r.get('objetivo', 100.0)
            ambiente.penalidade_passo = r.get('passo', -0.1)
            ambiente.penalidade_colisao = r.get('colisao', -5.0)
            ambiente.recompensa_aproximar = r.get('aproximar', 1.0)
            ambiente.penalidade_afastar = r.get('afastar', -1.0)
        
        # Gerar obstáculos
        ambiente._gerar_ambiente(config.get('obstaculos_fixos'))
        
        return ambiente
    
    def _gerar_ambiente(self, obstaculos_fixos: list = None) -> None:
        """Gera o ambiente com obstáculos."""
        # Colocar farol
        if self.objetivo:
            self.grid[self.objetivo[1], self.objetivo[0]] = TipoCelula.OBJETIVO
        
        # Colocar posição inicial
        if self.posicao_inicial:
            self.grid[self.posicao_inicial[1], self.posicao_inicial[0]] = TipoCelula.INICIO
        
        if obstaculos_fixos:
            # Usar obstáculos predefinidos
            for (x, y) in obstaculos_fixos:
                if (x, y) != self.objetivo and (x, y) != self.posicao_inicial:
                    self.grid[y, x] = TipoCelula.OBSTACULO
        else:
            # Gerar aleatoriamente
            self._gerar_obstaculos_aleatorios()
    
    def _gerar_obstaculos_aleatorios(self) -> None:
        """Gera obstáculos aleatoriamente com a densidade especificada."""
        num_obstaculos = int(self.largura * self.altura * self.densidade_obstaculos)
        
        celulas_livres = [
            (x, y) 
            for x in range(self.largura) 
            for y in range(self.altura)
            if self.grid[y, x] == TipoCelula.LIVRE
        ]
        
        # Garantir caminho livre (simples: não bloquear adjacentes ao início/fim)
        if self.posicao_inicial:
            celulas_livres = [
                c for c in celulas_livres 
                if abs(c[0] - self.posicao_inicial[0]) + abs(c[1] - self.posicao_inicial[1]) > 1
            ]
        if self.objetivo:
            celulas_livres = [
                c for c in celulas_livres 
                if abs(c[0] - self.objetivo[0]) + abs(c[1] - self.objetivo[1]) > 1
            ]
        
        obstaculos = random.sample(celulas_livres, min(num_obstaculos, len(celulas_livres)))
        for (x, y) in obstaculos:
            self.grid[y, x] = TipoCelula.OBSTACULO
    
    def observacaoPara(self, agente: Agente) -> Observacao:
        """Gera observação para o agente."""
        x, y = agente.posicao
        
        # Verificar se agente está iluminado pelo feixe
        iluminado = self._celula_iluminada(x, y)
        
        # Verificar iluminação das células adjacentes (dica direcional básica)
        vizinhos_iluminados = {}
        direcao_scores = {}
        
        if self.feixe_ativo:
            # Verificar com bounds checking
            vizinhos_iluminados = {
                'N': y > 0 and self._celula_iluminada(x, y - 1),
                'S': y < self.altura - 1 and self._celula_iluminada(x, y + 1),
                'E': x < self.largura - 1 and self._celula_iluminada(x + 1, y),
                'O': x > 0 and self._celula_iluminada(x - 1, y)
            }
            # Scores melhorados: verifica múltiplas células em cada direção
            direcao_scores = self.calcular_direcao_feixe_para_agente((x, y))
        
        return Observacao(
            posicao=(x, y),
            vizinhanca=self._obter_vizinhanca(x, y),
            direcao_objetivo=self._calcular_direcao_objetivo((x, y)),
            distancia_objetivo=self._distancia_objetivo((x, y)),
            dados_extra={
                'iluminado': iluminado, 
                'angulo_feixe': self.angulo_feixe,
                'vizinhos_iluminados': vizinhos_iluminados,
                'direcao_scores': direcao_scores  # Scores para cada direção
            }
        )
    
    def _celula_iluminada(self, x: int, y: int) -> bool:
        """
        Verifica se uma célula está dentro do feixe de luz do farol.
        
        O feixe é bloqueado por obstáculos - usa raycast do farol até à célula.
        """
        if not self.feixe_ativo or self.objetivo is None:
            return False
        
        farol_x, farol_y = self.objetivo
        
        # Distância da célula ao farol
        dx = x - farol_x
        dy = y - farol_y
        distancia = (dx**2 + dy**2) ** 0.5
        
        # Fora do raio ou na posição do farol
        if distancia > self.raio_feixe or distancia == 0:
            return False
        
        # Calcular ângulo da célula relativamente ao farol
        angulo_celula = math.degrees(math.atan2(-dy, dx))  # -dy porque y cresce para baixo
        angulo_celula = angulo_celula % 360
        
        # Verificar se está dentro do cone
        diff = abs(angulo_celula - self.angulo_feixe)
        if diff > 180:
            diff = 360 - diff
        
        if diff > self.abertura_feixe / 2:
            return False
        
        # Verificar se há obstáculo a bloquear a luz (raycast)
        if self._luz_bloqueada(farol_x, farol_y, x, y):
            return False
        
        return True
    
    def _luz_bloqueada(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        """
        Verifica se há obstáculo entre o farol (x0,y0) e a célula (x1,y1).
        Usa algoritmo de Bresenham para traçar linha.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        if dx > dy:
            err = dx / 2
            while x != x1:
                x += sx
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                # Verificar se esta célula intermédia é obstáculo
                if (x, y) != (x1, y1):  # Não contar a célula destino
                    if self._e_obstaculo(x, y):
                        return True
        else:
            err = dy / 2
            while y != y1:
                y += sy
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                # Verificar se esta célula intermédia é obstáculo
                if (x, y) != (x1, y1):
                    if self._e_obstaculo(x, y):
                        return True
        
        return False
    
    def _atualizar_feixe(self) -> None:
        """Roda o feixe de luz do farol."""
        self.angulo_feixe = (self.angulo_feixe + self.velocidade_rotacao) % 360
    
    def obter_celulas_iluminadas(self) -> list:
        """Retorna lista de células atualmente iluminadas pelo feixe."""
        celulas = []
        if not self.feixe_ativo:
            return celulas
        
        for x in range(self.largura):
            for y in range(self.altura):
                if self._celula_iluminada(x, y):
                    celulas.append((x, y))
        return celulas
    
    def calcular_direcao_feixe_para_agente(self, pos_agente: tuple) -> dict:
        """
        Calcula a direção provável do farol baseado em múltiplas células iluminadas.
        
        Retorna um dicionário com scores para cada direção (N, S, E, O).
        Score maior = mais provável que o farol esteja nessa direção.
        """
        x, y = pos_agente
        scores = {'N': 0, 'S': 0, 'E': 0, 'O': 0}
        
        if not self.feixe_ativo:
            return scores
        
        # Verificar várias células em cada direção (com bounds checking)
        raio_check = 3
        for r in range(1, raio_check + 1):
            # Norte
            if y - r >= 0 and self._celula_iluminada(x, y - r):
                scores['N'] += (raio_check - r + 1)
            # Sul
            if y + r < self.altura and self._celula_iluminada(x, y + r):
                scores['S'] += (raio_check - r + 1)
            # Este
            if x + r < self.largura and self._celula_iluminada(x + r, y):
                scores['E'] += (raio_check - r + 1)
            # Oeste
            if x - r >= 0 and self._celula_iluminada(x - r, y):
                scores['O'] += (raio_check - r + 1)
        
        return scores
    
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
    
    def atualizacao(self) -> None:
        """
        Atualiza o estado do ambiente após um passo.
        Roda o feixe de luz do farol.
        """
        super().atualizacao()
        self._atualizar_feixe()
    
    def _recompensa_colisao(self) -> float:
        """Recompensa por colidir."""
        return self.penalidade_colisao
    
    def reset(self) -> None:
        """Reinicia o ambiente."""
        super().reset()
        self._distancia_anterior.clear()
        self.angulo_feixe = 0.0  # Resetar ângulo do feixe
        
        # Regenerar obstáculos se necessário (opcional)
        # self._gerar_obstaculos_aleatorios()
    
    def visualizar_texto(self) -> str:
        """Retorna representação textual do ambiente."""
        simbolos = {
            TipoCelula.LIVRE: '.',
            TipoCelula.OBSTACULO: '#',
            TipoCelula.PAREDE: 'X',
            TipoCelula.OBJETIVO: 'F',  # Farol
            TipoCelula.INICIO: 'S'
        }
        
        linhas = []
        for y in range(self.altura):
            linha = ''
            for x in range(self.largura):
                # Verificar se há agente nesta posição
                agente_aqui = any(a.posicao == (x, y) for a in self.agentes)
                if agente_aqui:
                    linha += 'A'
                else:
                    linha += simbolos.get(self.grid[y, x], '?')
            linhas.append(linha)
        
        return '\n'.join(linhas)
