"""
Motor de Simulação - Controlador central do sistema SMA.
"""

import json
import time
from typing import List, Optional, Dict, Any, Type
from pathlib import Path

from .ambiente import Ambiente
from .agente import Agente
from .observacao import Observacao


class MotorDeSimulacao:
    """
    Controlador central do simulador de sistemas multi-agente.
    
    Gere o ciclo de tempo, a ordem de execução das ações dos agentes,
    e a sincronização entre agentes e ambiente.
    
    Attributes:
        ambiente: Ambiente atual da simulação
        agentes: Lista de agentes na simulação
        config: Configuração carregada do ficheiro
        episodio_atual: Número do episódio atual
        passo_atual: Passo atual dentro do episódio
    """
    
    def __init__(self):
        self.ambiente: Optional[Ambiente] = None
        self.agentes: List[Agente] = []
        self.config: Dict[str, Any] = {}
        
        # Estado da simulação
        self.episodio_atual: int = 0
        self.passo_atual: int = 0
        self.max_passos: int = 1000
        self.max_episodios: int = 100
        
        # Callbacks para visualização
        self._callback_passo: Optional[callable] = None
        self._callback_episodio: Optional[callable] = None
        
        # Métricas
        self.historico_recompensas: List[float] = []
        self.historico_passos: List[int] = []
        self.historico_sucesso: List[bool] = []
        
        # Controlo
        self._pausado = False
        self._a_correr = False
        self._delay_passo = 0.0  # Delay entre passos (para visualização)
    
    @classmethod
    def cria(cls, nome_ficheiro_parametros: str) -> 'MotorDeSimulacao':
        """
        Cria um motor de simulação a partir de um ficheiro de parâmetros.
        
        Args:
            nome_ficheiro_parametros: Caminho para o ficheiro JSON de configuração
            
        Returns:
            Nova instância do motor configurada
        """
        motor = cls()
        motor._carregar_configuracao(nome_ficheiro_parametros)
        return motor
    
    def _carregar_configuracao(self, caminho: str) -> None:
        """Carrega configuração de um ficheiro JSON."""
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"Ficheiro de configuração não encontrado: {caminho}")
        
        with open(caminho, 'r') as f:
            self.config = json.load(f)
        
        # Aplicar configurações
        self.max_passos = self.config.get('max_passos', 1000)
        self.max_episodios = self.config.get('max_episodios', 100)
        self._delay_passo = self.config.get('delay_passo', 0.0)
    
    def definir_ambiente(self, ambiente: Ambiente) -> None:
        """Define o ambiente da simulação."""
        self.ambiente = ambiente
    
    def adicionar_agente(self, agente: Agente) -> None:
        """Adiciona um agente à simulação."""
        self.agentes.append(agente)
        if self.ambiente:
            self.ambiente.adicionar_agente(agente)
    
    def listaAgentes(self) -> List[Agente]:
        """Retorna a lista de agentes na simulação."""
        return self.agentes.copy()
    
    def executa(self, modo_visual: bool = False) -> Dict[str, Any]:
        """
        Executa a simulação completa.
        
        Args:
            modo_visual: Se True, adiciona delays para visualização
            
        Returns:
            Dicionário com métricas da simulação
        """
        self._a_correr = True
        
        for episodio in range(self.max_episodios):
            if not self._a_correr:
                break
            
            self.episodio_atual = episodio
            resultado_episodio = self._executar_episodio(modo_visual)
            
            # Registar métricas
            self.historico_recompensas.append(resultado_episodio['recompensa_total'])
            self.historico_passos.append(resultado_episodio['passos'])
            self.historico_sucesso.append(resultado_episodio['sucesso'])
            
            # Callback de episódio
            if self._callback_episodio:
                self._callback_episodio(episodio, resultado_episodio)
        
        return self._compilar_metricas()
    
    def _executar_episodio(self, modo_visual: bool = False) -> Dict[str, Any]:
        """
        Executa um único episódio.
        
        Returns:
            Dicionário com resultados do episódio
        """
        # Reset
        self.ambiente.reset()
        self.passo_atual = 0
        recompensa_total = 0.0
        
        while self.passo_atual < self.max_passos and not self.ambiente.episodio_terminado:
            if not self._a_correr:
                break
            
            while self._pausado:
                time.sleep(0.1)
            
            # Executar um passo
            recompensa_passo = self._executar_passo()
            recompensa_total += recompensa_passo
            self.passo_atual += 1
            
            # Visualização
            if modo_visual and self._delay_passo > 0:
                time.sleep(self._delay_passo)
            
            # Callback de passo
            if self._callback_passo:
                self._callback_passo(self.passo_atual, recompensa_passo)
        
        return {
            'passos': self.passo_atual,
            'recompensa_total': recompensa_total,
            'sucesso': self.ambiente.episodio_terminado
        }
    
    def _executar_passo(self) -> float:
        """
        Executa um passo da simulação (modo síncrono).
        
        Returns:
            Recompensa total do passo
        """
        recompensa_total = 0.0
        
        for agente in self.agentes:
            # 1. Gerar observação para o agente
            obs = self.ambiente.observacaoPara(agente)
            agente.observacao(obs)
            
            # 2. Agente decide ação
            accao = agente.age()
            
            # 3. Executar ação no ambiente
            sucesso, recompensa = self.ambiente.agir(accao, agente)
            
            # 4. Dar feedback ao agente
            agente.avaliacaoEstadoAtual(recompensa)
            recompensa_total += recompensa
        
        # 5. Atualizar ambiente
        self.ambiente.atualizacao()
        
        return recompensa_total
    
    def _executar_passo_threads(self) -> float:
        """
        Executa um passo da simulação com agentes como threads.
        Os agentes processam observações em paralelo.
        
        Returns:
            Recompensa total do passo
        """
        recompensa_total = 0.0
        
        # 1. Enviar observações para todos os agentes
        for agente in self.agentes:
            obs = self.ambiente.observacaoPara(agente)
            agente.observacao(obs)
        
        # 2. Esperar ações de todos os agentes (paralelo)
        accoes = {}
        for agente in self.agentes:
            accao = agente.esperar_accao(timeout=5.0)
            if accao:
                accoes[agente] = accao
        
        # 3. Executar ações no ambiente (sequencial para evitar conflitos)
        for agente, accao in accoes.items():
            sucesso, recompensa = self.ambiente.agir(accao, agente)
            agente.avaliacaoEstadoAtual(recompensa)
            recompensa_total += recompensa
        
        # 4. Atualizar ambiente
        self.ambiente.atualizacao()
        
        return recompensa_total
    
    def iniciar_agentes_threads(self) -> None:
        """Inicia todos os agentes como threads."""
        for agente in self.agentes:
            agente.iniciar_thread()
    
    def parar_agentes_threads(self) -> None:
        """Para todas as threads dos agentes."""
        for agente in self.agentes:
            agente.parar()
    
    def registar_agentes_entre_si(self) -> None:
        """Regista cada agente com conhecimento dos outros (para comunicação)."""
        for agente in self.agentes:
            for outro in self.agentes:
                if agente != outro:
                    agente.registar_agente(outro)
    
    def _compilar_metricas(self) -> Dict[str, Any]:
        """Compila métricas finais da simulação."""
        if not self.historico_recompensas:
            return {}
        
        return {
            'episodios_totais': len(self.historico_recompensas),
            'recompensa_media': sum(self.historico_recompensas) / len(self.historico_recompensas),
            'passos_medios': sum(self.historico_passos) / len(self.historico_passos),
            'taxa_sucesso': sum(self.historico_sucesso) / len(self.historico_sucesso),
            'historico_recompensas': self.historico_recompensas,
            'historico_passos': self.historico_passos,
            'historico_sucesso': self.historico_sucesso
        }
    
    # Controlo de execução
    def pausar(self) -> None:
        """Pausa a simulação."""
        self._pausado = True
    
    def continuar(self) -> None:
        """Continua a simulação."""
        self._pausado = False
    
    def parar(self) -> None:
        """Para a simulação."""
        self._a_correr = False
        for agente in self.agentes:
            agente.parar()
    
    def definir_callback_passo(self, callback: callable) -> None:
        """Define callback chamado a cada passo."""
        self._callback_passo = callback
    
    def definir_callback_episodio(self, callback: callable) -> None:
        """Define callback chamado a cada episódio."""
        self._callback_episodio = callback
    
    def definir_delay(self, delay: float) -> None:
        """Define delay entre passos (para visualização)."""
        self._delay_passo = delay
