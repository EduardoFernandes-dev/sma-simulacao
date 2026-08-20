"""
Métricas e Análise de Resultados.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime


class MetricasSimulacao:
    """
    Classe para registo e análise de métricas da simulação.
    """
    
    def __init__(self):
        self.recompensas: List[float] = []
        self.passos: List[int] = []
        self.sucessos: List[bool] = []
        self.epsilon_historico: List[float] = []
        
        # Metadata
        self.nome_experiencia = "simulacao"
        self.inicio = datetime.now()
    
    def registar_episodio(self, recompensa: float, passos: int, 
                          sucesso: bool, epsilon: float = None) -> None:
        """Regista métricas de um episódio."""
        self.recompensas.append(recompensa)
        self.passos.append(passos)
        self.sucessos.append(sucesso)
        if epsilon is not None:
            self.epsilon_historico.append(epsilon)
    
    def media_movel(self, dados: List[float], janela: int = 50) -> List[float]:
        """Calcula média móvel dos dados."""
        if len(dados) < janela:
            return dados
        
        return [
            np.mean(dados[max(0, i-janela):i+1])
            for i in range(len(dados))
        ]
    
    def estatisticas(self) -> Dict:
        """Retorna estatísticas agregadas."""
        if not self.recompensas:
            return {}
        
        return {
            'episodios_totais': len(self.recompensas),
            'recompensa_media': np.mean(self.recompensas),
            'recompensa_std': np.std(self.recompensas),
            'recompensa_max': max(self.recompensas),
            'recompensa_min': min(self.recompensas),
            'passos_medios': np.mean(self.passos),
            'taxa_sucesso': sum(self.sucessos) / len(self.sucessos),
            'ultimos_50_recompensa': np.mean(self.recompensas[-50:]) if len(self.recompensas) >= 50 else np.mean(self.recompensas),
            'ultimos_50_sucesso': sum(self.sucessos[-50:]) / min(50, len(self.sucessos))
        }
    
    def plotar_curva_aprendizagem(self, caminho: str = None, 
                                   mostrar: bool = True) -> None:
        """Gera gráfico da curva de aprendizagem."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Curva de Aprendizagem - {self.nome_experiencia}", fontsize=14)
        
        # Recompensa por episódio
        ax1 = axes[0, 0]
        ax1.plot(self.recompensas, alpha=0.3, label='Recompensa')
        ax1.plot(self.media_movel(self.recompensas), label='Média Móvel (50)', linewidth=2)
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Recompensa')
        ax1.set_title('Recompensa por Episódio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Passos por episódio
        ax2 = axes[0, 1]
        ax2.plot(self.passos, alpha=0.3, label='Passos')
        ax2.plot(self.media_movel([float(p) for p in self.passos]), 
                 label='Média Móvel (50)', linewidth=2)
        ax2.set_xlabel('Episódio')
        ax2.set_ylabel('Passos')
        ax2.set_title('Passos por Episódio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Taxa de sucesso
        ax3 = axes[1, 0]
        taxa_sucesso = [sum(self.sucessos[:i+1])/(i+1) for i in range(len(self.sucessos))]
        ax3.plot(taxa_sucesso, linewidth=2)
        ax3.set_xlabel('Episódio')
        ax3.set_ylabel('Taxa de Sucesso')
        ax3.set_title('Taxa de Sucesso Acumulada')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Epsilon (se disponível)
        ax4 = axes[1, 1]
        if self.epsilon_historico:
            ax4.plot(self.epsilon_historico, linewidth=2, color='orange')
            ax4.set_xlabel('Episódio')
            ax4.set_ylabel('Epsilon')
            ax4.set_title('Decaimento do Epsilon')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Sem dados de Epsilon', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Epsilon')
        
        plt.tight_layout()
        
        if caminho:
            plt.savefig(caminho, dpi=150, bbox_inches='tight')
            print(f"Gráfico guardado em: {caminho}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
    
    def guardar_csv(self, caminho: str) -> None:
        """Guarda métricas em CSV."""
        import csv
        
        caminho = Path(caminho)
        with open(caminho, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episodio', 'recompensa', 'passos', 'sucesso', 'epsilon'])
            
            for i in range(len(self.recompensas)):
                epsilon = self.epsilon_historico[i] if i < len(self.epsilon_historico) else ''
                writer.writerow([
                    i + 1,
                    self.recompensas[i],
                    self.passos[i],
                    int(self.sucessos[i]),
                    epsilon
                ])
        
        print(f"Métricas guardadas em: {caminho}")
    
    def imprimir_resumo(self) -> None:
        """Imprime resumo das métricas."""
        stats = self.estatisticas()
        if not stats:
            print("Sem dados disponíveis.")
            return
        
        print("\n" + "="*50)
        print("RESUMO DA SIMULAÇÃO")
        print("="*50)
        print(f"Episódios totais: {stats['episodios_totais']}")
        print(f"Recompensa média: {stats['recompensa_media']:.2f} ± {stats['recompensa_std']:.2f}")
        print(f"Passos médios: {stats['passos_medios']:.1f}")
        print(f"Taxa de sucesso: {stats['taxa_sucesso']*100:.1f}%")
        print(f"Últimos 50 eps - Recompensa: {stats['ultimos_50_recompensa']:.2f}")
        print(f"Últimos 50 eps - Sucesso: {stats['ultimos_50_sucesso']*100:.1f}%")
        print("="*50 + "\n")
