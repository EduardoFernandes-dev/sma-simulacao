"""
Classe Observação - Representa o que um agente percebe do ambiente.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class Observacao:
    posicao: Tuple[int, int]
    vizinhanca: Dict[str, str] = field(default_factory=dict)  # {'N': 'livre', 'S': 'obstaculo', ...}
    direcao_objetivo: Optional[str] = None  # 'N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'
    distancia_objetivo: Optional[float] = None
    dados_extra: Dict[str, Any] = field(default_factory=dict)
    
    def para_estado(self) -> Tuple:
        # Estado básico: posição + vizinhança
        vizinhanca_tuple = tuple(sorted(self.vizinhanca.items()))
        
        # Incluir vizinhos iluminados se disponível (dica direcional do farol)
        vizinhos_iluminados = self.dados_extra.get('vizinhos_iluminados', {})
        if vizinhos_iluminados:
            iluminados_tuple = tuple(sorted(vizinhos_iluminados.items()))
            return (self.posicao, vizinhanca_tuple, self.direcao_objetivo, iluminados_tuple)
        
        return (self.posicao, vizinhanca_tuple, self.direcao_objetivo)
    
    def para_array(self) -> np.ndarray:
        # Encoding básico
        features = list(self.posicao)
        
        # Adicionar vizinhança (1 = livre, 0 = obstáculo/parede)
        direcoes = ['N', 'S', 'E', 'O']
        for d in direcoes:
            estado = self.vizinhanca.get(d, 'desconhecido')
            features.append(1 if estado == 'livre' else 0)
        
        return np.array(features, dtype=np.float32)
    
    @classmethod
    def from_sensor_data(cls, dados_sensor: dict) -> 'Observacao':
        """
        Cria uma Observação a partir dos dados recolhidos pelos sensores.
        
        Args:
            dados_sensor: Dicionário com dados de cada sensor
                         Ex: {'Posicao': (5,5), 'Vizinhanca': {...}, ...}
        """
        # Extrair dados de sensores individuais
        posicao = dados_sensor.get('Posicao', (0, 0))
        vizinhanca = dados_sensor.get('Vizinhanca', {})
        direcao = dados_sensor.get('DirecaoObjetivo', None)
        distancia = dados_sensor.get('Distancia', None)
        
        # Dados de iluminação (se existir)
        iluminacao = dados_sensor.get('Iluminacao', {})
        
        # Para sensor composto (FarolCompleto, etc)
        if len(dados_sensor) == 1:
            # É um sensor composto, extrair do dict interno
            sensor_nome = list(dados_sensor.keys())[0]
            dados_internos = dados_sensor[sensor_nome]
            if isinstance(dados_internos, dict):
                posicao = dados_internos.get('Posicao', posicao)
                vizinhanca = dados_internos.get('Vizinhanca', vizinhanca)
                direcao = dados_internos.get('DirecaoObjetivo', direcao)
                distancia = dados_internos.get('Distancia', distancia)
                iluminacao = dados_internos.get('Iluminacao', iluminacao)
        
        dados_extra = {}
        if iluminacao:
            dados_extra['iluminado'] = iluminacao.get('iluminado', False)
            dados_extra['vizinhos_iluminados'] = iluminacao.get('vizinhos_iluminados', {})
            dados_extra['direcao_scores'] = iluminacao.get('direcao_scores', {})
        
        return cls(
            posicao=posicao,
            vizinhanca=vizinhanca,
            direcao_objetivo=direcao,
            distancia_objetivo=distancia,
            dados_extra=dados_extra
        )
    
    def __str__(self) -> str:
        return f"Obs(pos={self.posicao}, dir={self.direcao_objetivo})"
