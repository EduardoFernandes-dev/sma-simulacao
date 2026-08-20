"""
Sensores para o Simulador Multi-Agente.

Os sensores seguem o padrão Strategy - podem ser trocados sem alterar o código do agente.
Cada sensor filtra a "verdade absoluta" do ambiente, definindo o que o agente pode perceber.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.agente import Agente
    from src.core.ambiente import Ambiente


class Sensor(ABC):
    """Classe abstrata base para todos os sensores."""
    
    def __init__(self, nome: str = "SensorBase"):
        self.nome = nome
        self.agente: 'Agente' = None
        self.ambiente: 'Ambiente' = None
    
    @abstractmethod
    def perceber(self) -> Any:
        """Realiza a perceção e retorna os dados sensoriais."""
        pass
    
    def configurar(self, ambiente: 'Ambiente') -> None:
        """Configura o sensor com referência ao ambiente."""
        self.ambiente = ambiente
    
    def __str__(self) -> str:
        return f"Sensor({self.nome})"


class SensorDirecaoObjetivo(Sensor):
    """
    Sensor que detecta a direção relativa ao objetivo.
    Retorna: 'N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'
    """
    
    def __init__(self):
        super().__init__("DirecaoObjetivo")
    
    def perceber(self) -> Optional[str]:
        if self.ambiente is None or self.agente is None:
            return None
        return self.ambiente._calcular_direcao_objetivo(self.agente.posicao)


class SensorVizinhanca(Sensor):
    """
    Sensor que detecta o estado das células vizinhas (N, S, E, O).
    Retorna: {'N': 'livre'/'obstaculo'/'parede', 'S': ..., 'E': ..., 'O': ...}
    """
    
    def __init__(self, incluir_diagonais: bool = False):
        super().__init__("Vizinhanca")
        self.incluir_diagonais = incluir_diagonais
    
    def perceber(self) -> Dict[str, str]:
        if self.ambiente is None or self.agente is None:
            return {}
        
        x, y = self.agente.posicao
        vizinhanca = self.ambiente._obter_vizinhanca(x, y)
        
        if self.incluir_diagonais:
            diagonais = {'NE': (1, -1), 'SE': (1, 1), 'SO': (-1, 1), 'NO': (-1, -1)}
            for direcao, (dx, dy) in diagonais.items():
                nx, ny = x + dx, y + dy
                if self.ambiente._dentro_limites(nx, ny):
                    vizinhanca[direcao] = self.ambiente.grid[ny, nx]
                else:
                    vizinhanca[direcao] = 'parede'
        
        return vizinhanca


class SensorDistancia(Sensor):
    """
    Sensor que detecta a distância ao objetivo (Manhattan).
    Retorna: float (distância em células)
    """
    
    def __init__(self):
        super().__init__("Distancia")
    
    def perceber(self) -> float:
        if self.ambiente is None or self.agente is None:
            return float('inf')
        return self.ambiente._distancia_objetivo(self.agente.posicao)


class SensorIluminacao(Sensor):
    """
    Sensor que detecta se o agente está iluminado pelo feixe do farol.
    Específico para o AmbienteFarol.
    Retorna: {'iluminado': bool, 'vizinhos_iluminados': dict, 'direcao_scores': dict}
    """
    
    def __init__(self):
        super().__init__("Iluminacao")
    
    def perceber(self) -> Dict:
        if self.ambiente is None or self.agente is None:
            return {'iluminado': False, 'vizinhos_iluminados': {}, 'direcao_scores': {}}
        
        # Verificar se é AmbienteFarol
        if not hasattr(self.ambiente, 'feixe_ativo'):
            return {'iluminado': False, 'vizinhos_iluminados': {}, 'direcao_scores': {}}
        
        x, y = self.agente.posicao
        
        if not self.ambiente.feixe_ativo:
            return {'iluminado': False, 'vizinhos_iluminados': {}, 'direcao_scores': {}}
        
        iluminado = self.ambiente._celula_iluminada(x, y)
        
        vizinhos_iluminados = {
            'N': y > 0 and self.ambiente._celula_iluminada(x, y - 1),
            'S': y < self.ambiente.altura - 1 and self.ambiente._celula_iluminada(x, y + 1),
            'E': x < self.ambiente.largura - 1 and self.ambiente._celula_iluminada(x + 1, y),
            'O': x > 0 and self.ambiente._celula_iluminada(x - 1, y)
        }
        
        direcao_scores = self.ambiente.calcular_direcao_feixe_para_agente((x, y))
        
        return {
            'iluminado': iluminado,
            'vizinhos_iluminados': vizinhos_iluminados,
            'direcao_scores': direcao_scores
        }


class SensorPosicao(Sensor):
    """
    Sensor que retorna a posição atual do agente.
    Retorna: (x, y)
    """
    
    def __init__(self):
        super().__init__("Posicao")
    
    def perceber(self) -> tuple:
        if self.agente is None:
            return (0, 0)
        return self.agente.posicao


class SensorComposto(Sensor):
    """
    Sensor que combina múltiplos sensores num único.
    Útil para criar configurações de sensores pré-definidas.
    """
    
    def __init__(self, nome: str, sensores: list):
        super().__init__(nome)
        self.sensores = sensores
    
    def configurar(self, ambiente: 'Ambiente') -> None:
        super().configurar(ambiente)
        for sensor in self.sensores:
            sensor.configurar(ambiente)
    
    def perceber(self) -> Dict[str, Any]:
        resultado = {}
        for sensor in self.sensores:
            sensor.agente = self.agente
            resultado[sensor.nome] = sensor.perceber()
        return resultado


# Configurações de sensores pré-definidas
def criar_sensor_farol_completo() -> SensorComposto:
    """Sensor completo para o ambiente Farol (visão global)."""
    return SensorComposto("FarolCompleto", [
        SensorPosicao(),
        SensorVizinhanca(),
        SensorDirecaoObjetivo(),
        SensorDistancia(),
        SensorIluminacao()
    ])


def criar_sensor_farol_limitado() -> SensorComposto:
    """Sensor limitado para o ambiente Farol (só vizinhança e direção)."""
    return SensorComposto("FarolLimitado", [
        SensorVizinhanca(),
        SensorDirecaoObjetivo()
    ])


def criar_sensor_labirinto() -> SensorComposto:
    """Sensor para o ambiente Labirinto (com direção ao objetivo)."""
    return SensorComposto("Labirinto", [
        SensorPosicao(),
        SensorVizinhanca(incluir_diagonais=False),
        SensorDirecaoObjetivo()
    ])
