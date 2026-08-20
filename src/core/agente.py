"""
Classe Abstrata Agente - Interface base para todos os agentes.
Implementa comunicação entre agentes e execução como Thread.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING, Dict, Any
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from datetime import datetime

if TYPE_CHECKING:
    from .observacao import Observacao
    from .accao import Accao
    from src.sensores.sensor_base import Sensor


@dataclass
class Mensagem:
    """Representa uma mensagem trocada entre agentes."""
    conteudo: str
    remetente: str
    timestamp: datetime
    tipo: str = "info"  # info, pedido, resposta, alerta
    dados: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dados is None:
            self.dados = {}


class Agente(ABC, threading.Thread):
    """
    Classe abstrata base para todos os agentes.
    
    Suporta:
    - Execução como Thread independente
    - Comunicação assíncrona entre agentes via fila de mensagens
    - Sensores modulares (padrão Strategy)
    """
    
    def __init__(self, nome: str, posicao: tuple = (0, 0)):
        threading.Thread.__init__(self, daemon=True)
        self.nome = nome
        self.posicao = posicao
        self.sensores: List['Sensor'] = []
        self.recompensa_acumulada: float = 0.0
        self.modo_aprendizagem: bool = True
        
        # Controlo de execução da Thread
        self._running = False
        self._observacao_atual: Optional['Observacao'] = None
        self._ultima_recompensa: float = 0.0
        
        # Sincronização com o simulador
        self._evento_observacao = threading.Event()
        self._evento_accao_processada = threading.Event()
        self._proxima_accao: Optional['Accao'] = None
        self._lock = threading.Lock()
        
        # Sistema de comunicação entre agentes
        self._caixa_mensagens: Queue[Mensagem] = Queue()
        self._agentes_conhecidos: Dict[str, 'Agente'] = {}
        self._historico_mensagens: List[Mensagem] = []
    
    @classmethod
    def cria(cls, nome_ficheiro_parametros: str) -> 'Agente':
        import json
        with open(nome_ficheiro_parametros, 'r') as f:
            config = json.load(f)
        raise NotImplementedError("Subclasses devem implementar cria()")
    
    # =========== Interface principal ===========
    
    def observacao(self, obs: 'Observacao') -> None:
        """Recebe uma observação do ambiente."""
        with self._lock:
            self._observacao_atual = obs
        self._evento_observacao.set()
    
    @abstractmethod
    def age(self) -> 'Accao':
        """Decide e retorna a próxima ação a executar."""
        pass
    
    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        """Recebe feedback sobre a última ação."""
        with self._lock:
            self._ultima_recompensa = recompensa
            self.recompensa_acumulada += recompensa
    
    def instala(self, sensor: 'Sensor') -> None:
        """Instala um sensor no agente."""
        self.sensores.append(sensor)
        sensor.agente = self
    
    def perceber(self) -> dict:
        """Recolhe dados de todos os sensores instalados."""
        dados = {}
        for sensor in self.sensores:
            dados[sensor.nome] = sensor.perceber()
        return dados
    
    # =========== Sistema de Comunicação ===========
    
    def comunica(self, mensagem: str, de_agente: 'Agente') -> None:
        """
        Recebe uma mensagem de outro agente.
        A mensagem é colocada na fila para processamento assíncrono.
        
        Args:
            mensagem: Conteúdo da mensagem
            de_agente: Agente que enviou a mensagem
        """
        msg = Mensagem(
            conteudo=mensagem,
            remetente=de_agente.nome,
            timestamp=datetime.now()
        )
        self._caixa_mensagens.put(msg)
        self._historico_mensagens.append(msg)
    
    def enviar_mensagem(self, destinatario: 'Agente', mensagem: str, 
                        tipo: str = "info", dados: Dict[str, Any] = None) -> bool:
        """
        Envia uma mensagem para outro agente.
        
        Args:
            destinatario: Agente que vai receber a mensagem
            mensagem: Conteúdo da mensagem
            tipo: Tipo de mensagem (info, pedido, resposta, alerta)
            dados: Dados adicionais estruturados
            
        Returns:
            True se a mensagem foi enviada com sucesso
        """
        if destinatario is None:
            return False
        
        msg = Mensagem(
            conteudo=mensagem,
            remetente=self.nome,
            timestamp=datetime.now(),
            tipo=tipo,
            dados=dados or {}
        )
        destinatario._caixa_mensagens.put(msg)
        destinatario._historico_mensagens.append(msg)
        return True
    
    def broadcast(self, mensagem: str, tipo: str = "info") -> int:
        """
        Envia mensagem para todos os agentes conhecidos.
        
        Returns:
            Número de agentes que receberam a mensagem
        """
        count = 0
        for agente in self._agentes_conhecidos.values():
            if self.enviar_mensagem(agente, mensagem, tipo):
                count += 1
        return count
    
    def ler_mensagens(self, bloquear: bool = False, timeout: float = 0.1) -> List[Mensagem]:
        """
        Lê todas as mensagens pendentes na caixa de correio.
        
        Args:
            bloquear: Se True, espera até haver mensagens
            timeout: Tempo máximo de espera (se bloquear=True)
            
        Returns:
            Lista de mensagens recebidas
        """
        mensagens = []
        
        if bloquear:
            try:
                msg = self._caixa_mensagens.get(timeout=timeout)
                mensagens.append(msg)
            except Empty:
                pass
        
        # Ler todas as mensagens restantes
        while not self._caixa_mensagens.empty():
            try:
                mensagens.append(self._caixa_mensagens.get_nowait())
            except Empty:
                break
        
        return mensagens
    
    def tem_mensagens(self) -> bool:
        """Verifica se há mensagens pendentes."""
        return not self._caixa_mensagens.empty()
    
    def registar_agente(self, agente: 'Agente') -> None:
        """Regista um agente conhecido para comunicação."""
        self._agentes_conhecidos[agente.nome] = agente
    
    def processar_mensagens(self) -> None:
        """
        Processa mensagens pendentes.
        Subclasses podem sobrescrever para implementar lógica específica.
        """
        mensagens = self.ler_mensagens()
        for msg in mensagens:
            self._processar_mensagem(msg)
    
    def _processar_mensagem(self, msg: Mensagem) -> None:
        """
        Processa uma mensagem individual.
        Subclasses podem sobrescrever para comportamento específico.
        """
        pass  # Implementação default não faz nada
    
    # =========== Execução como Thread ===========
    
    def run(self) -> None:
        """
        Loop principal da Thread do agente.
        Espera por observações, processa mensagens, decide ações.
        """
        self._running = True
        
        while self._running:
            # Esperar por observação do simulador
            observacao_recebida = self._evento_observacao.wait(timeout=0.1)
            
            if not self._running:
                break
            
            if observacao_recebida:
                self._evento_observacao.clear()
                
                # Processar mensagens pendentes
                self.processar_mensagens()
                
                # Decidir ação
                with self._lock:
                    self._proxima_accao = self.age()
                
                # Sinalizar que ação está pronta
                self._evento_accao_processada.set()
    
    def iniciar_thread(self) -> None:
        """Inicia o agente como thread."""
        if not self.is_alive():
            self._running = True
            self.start()
    
    def parar(self) -> None:
        """Para a execução da thread do agente."""
        self._running = False
        self._evento_observacao.set()  # Desbloquear se estiver à espera
    
    def esperar_accao(self, timeout: float = 5.0) -> Optional['Accao']:
        """
        Espera que o agente tenha uma ação pronta.
        Usado pelo simulador em modo thread.
        """
        if self._evento_accao_processada.wait(timeout=timeout):
            self._evento_accao_processada.clear()
            with self._lock:
                return self._proxima_accao
        return None
    
    # =========== Outros métodos ===========
    
    def reset(self, posicao: tuple = None) -> None:
        """Reset do estado do agente para novo episódio."""
        if posicao:
            self.posicao = posicao
        self.recompensa_acumulada = 0.0
        self._observacao_atual = None
        self._ultima_recompensa = 0.0
        self._evento_observacao.clear()
        self._evento_accao_processada.clear()
        # Limpar mensagens pendentes
        while not self._caixa_mensagens.empty():
            try:
                self._caixa_mensagens.get_nowait()
            except Empty:
                break
    
    def definir_modo(self, aprendizagem: bool) -> None:
        """Define se o agente está em modo de aprendizagem."""
        self.modo_aprendizagem = aprendizagem
    
    def estatisticas_comunicacao(self) -> Dict[str, Any]:
        """Retorna estatísticas de comunicação."""
        return {
            'mensagens_recebidas': len(self._historico_mensagens),
            'mensagens_pendentes': self._caixa_mensagens.qsize(),
            'agentes_conhecidos': list(self._agentes_conhecidos.keys())
        }
    
    def __str__(self) -> str:
        return f"Agente({self.nome}, pos={self.posicao})"
