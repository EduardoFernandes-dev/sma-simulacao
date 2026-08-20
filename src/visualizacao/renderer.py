"""
Renderer PyGame - Visualização da simulação.
"""

import pygame
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.ambiente import Ambiente
    from src.core.simulador import MotorDeSimulacao


# Cores
CORES = {
    'fundo': (30, 30, 30),
    'grid': (50, 50, 50),
    'livre': (60, 60, 60),
    'obstaculo': (100, 40, 40),
    'parede': (80, 80, 80),
    'objetivo': (50, 200, 50),
    'inicio': (50, 100, 200),
    'agente': (255, 200, 50),
    'texto': (255, 255, 255),
    'texto_dim': (150, 150, 150),
    'iluminado': (255, 255, 100)  # Amarelo claro para feixe
}


class Renderer:
    """
    Renderizador PyGame para visualização da simulação.
    
    Attributes:
        largura_janela: Largura da janela em pixels
        altura_janela: Altura da janela em pixels
        tamanho_celula: Tamanho de cada célula em pixels
    """
    
    def __init__(self, largura: int = 800, altura: int = 600):
        self.largura_janela = largura
        self.altura_janela = altura
        self.tamanho_celula = 40
        
        # PyGame
        self._screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._font_pequena: Optional[pygame.font.Font] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._inicializado = False
        
        # Estado
        self.ambiente: Optional['Ambiente'] = None
        self.simulador: Optional['MotorDeSimulacao'] = None
        
        # UI
        self.mostrar_info = True
        self.fps = 30
    
    def inicializar(self) -> None:
        """Inicializa o PyGame e cria a janela."""
        if self._inicializado:
            return
        
        pygame.init()
        pygame.display.set_caption("Simulador SMA - Agentes Autónomos")
        
        self._screen = pygame.display.set_mode((self.largura_janela, self.altura_janela))
        self._font = pygame.font.Font(None, 28)
        self._font_pequena = pygame.font.Font(None, 20)
        self._clock = pygame.time.Clock()
        self._inicializado = True
    
    def configurar(self, ambiente: 'Ambiente', simulador: 'MotorDeSimulacao' = None) -> None:
        """Configura o renderer com ambiente e simulador."""
        self.ambiente = ambiente
        self.simulador = simulador
        
        # Ajustar tamanho da célula para caber na janela
        margem = 200  # Espaço para informações
        max_largura = (self.largura_janela - margem) // ambiente.largura
        max_altura = self.altura_janela // ambiente.altura
        self.tamanho_celula = min(max_largura, max_altura, 50)
    
    def processar_eventos(self) -> bool:
        """
        Processa eventos PyGame.
        
        Returns:
            False se deve fechar, True caso contrário
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE and self.simulador:
                    # Toggle pause
                    if self.simulador._pausado:
                        self.simulador.continuar()
                    else:
                        self.simulador.pausar()
                elif event.key == pygame.K_i:
                    self.mostrar_info = not self.mostrar_info
        
        return True
    
    def renderizar(self) -> None:
        """Renderiza um frame da simulação."""
        if not self._inicializado or self.ambiente is None:
            return
        
        # Limpar ecrã
        self._screen.fill(CORES['fundo'])
        
        # Desenhar grid
        self._desenhar_grid()
        
        # Desenhar agentes
        self._desenhar_agentes()
        
        # Desenhar informações
        if self.mostrar_info:
            self._desenhar_info()
        
        # Atualizar display
        pygame.display.flip()
        self._clock.tick(self.fps)
    
    def _desenhar_grid(self) -> None:
        """Desenha a grelha do ambiente."""
        offset_x = 20
        offset_y = 20
        
        # Obter células iluminadas se ambiente for Farol
        celulas_iluminadas = set()
        if hasattr(self.ambiente, 'obter_celulas_iluminadas'):
            celulas_iluminadas = set(self.ambiente.obter_celulas_iluminadas())
        
        for y in range(self.ambiente.altura):
            for x in range(self.ambiente.largura):
                rect = pygame.Rect(
                    offset_x + x * self.tamanho_celula,
                    offset_y + y * self.tamanho_celula,
                    self.tamanho_celula - 2,
                    self.tamanho_celula - 2
                )
                
                # Cor baseada no tipo de célula
                tipo = self.ambiente.grid[y, x]
                if tipo == 'objetivo':
                    cor = CORES['objetivo']
                elif tipo == 'obstaculo':
                    cor = CORES['obstaculo']
                elif tipo == 'parede':
                    cor = CORES['parede']
                elif tipo == 'inicio':
                    cor = CORES['inicio']
                else:
                    cor = CORES['livre']
                
                pygame.draw.rect(self._screen, cor, rect)
                
                # Sobrepor cor de iluminação se célula está no feixe
                if (x, y) in celulas_iluminadas:
                    # Criar superfície semi-transparente
                    s = pygame.Surface((self.tamanho_celula - 2, self.tamanho_celula - 2))
                    s.set_alpha(80)  # Transparência
                    s.fill(CORES['iluminado'])
                    self._screen.blit(s, (rect.x, rect.y))
                
                pygame.draw.rect(self._screen, CORES['grid'], rect, 1)
    
    def _desenhar_agentes(self) -> None:
        """Desenha os agentes no ambiente."""
        offset_x = 20
        offset_y = 20
        
        for agente in self.ambiente.agentes:
            x, y = agente.posicao
            
            centro_x = offset_x + x * self.tamanho_celula + self.tamanho_celula // 2
            centro_y = offset_y + y * self.tamanho_celula + self.tamanho_celula // 2
            raio = self.tamanho_celula // 3
            
            # Círculo do agente
            pygame.draw.circle(self._screen, CORES['agente'], (centro_x, centro_y), raio)
            pygame.draw.circle(self._screen, (200, 150, 0), (centro_x, centro_y), raio, 2)
    
    def _desenhar_info(self) -> None:
        """Desenha painel de informações."""
        offset_x = self.ambiente.largura * self.tamanho_celula + 50
        y = 30
        espacamento = 25
        
        info = [
            f"Episódio: {self.simulador.episodio_atual if self.simulador else 0}",
            f"Passo: {self.ambiente.passos_episodio}",
            "",
            "Agentes:",
        ]
        
        for agente in self.ambiente.agentes:
            info.append(f"  {agente.nome}")
            info.append(f"    Pos: {agente.posicao}")
            info.append(f"    Rew: {agente.recompensa_acumulada:.1f}")
        
        info.extend([
            "",
            "Controlos:",
            "  ESPAÇO: Pausar",
            "  I: Toggle Info",
            "  ESC: Sair"
        ])
        
        for texto in info:
            if texto == "":
                y += espacamento // 2
                continue
            
            cor = CORES['texto'] if not texto.startswith("  ") else CORES['texto_dim']
            superficie = self._font_pequena.render(texto, True, cor)
            self._screen.blit(superficie, (offset_x, y))
            y += espacamento
    
    def fechar(self) -> None:
        """Fecha o PyGame."""
        if self._inicializado:
            pygame.quit()
            self._inicializado = False
