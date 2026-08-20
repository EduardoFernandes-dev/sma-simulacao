#!/usr/bin/env python3
"""
Comparação de Algoritmos Q-Learning vs SARSA.

Executa múltiplas runs de cada algoritmo e gera gráficos comparativos.

Uso:
    python comparar_algoritmos.py --ambiente farol --episodios 500 --runs 3
    python comparar_algoritmos.py --ambiente labirinto --episodios 1000 --runs 5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.ambientes.farol import AmbienteFarol
from src.ambientes.labirinto import AmbienteLabirinto
from src.agentes.agente_qlearning import AgenteQLearning
from src.agentes.agente_sarsa import AgenteSARSA
from src.utils.metricas import MetricasSimulacao


def criar_ambiente(tipo: str):
    """Cria ambiente para comparação."""
    if tipo == 'farol':
        ambiente = AmbienteFarol(largura=10, altura=10, densidade_obstaculos=0.1)
        ambiente.objetivo = (9, 9)
        ambiente.posicao_inicial = (0, 0)
        ambiente._gerar_ambiente()
        return ambiente
    elif tipo == 'labirinto':
        ambiente = AmbienteLabirinto(largura=11, altura=11)
        ambiente._gerar_labirinto_dfs()
        return ambiente
    raise ValueError(f"Ambiente desconhecido: {tipo}")


def executar_treino(ambiente, agente, episodios: int) -> MetricasSimulacao:
    """Executa treino e retorna métricas."""
    metricas = MetricasSimulacao()
    
    for episodio in range(episodios):
        ambiente.reset()
        recompensa_total = 0
        passos = 0
        
        while not ambiente.episodio_terminado and passos < 200:
            obs = ambiente.observacaoPara(agente)
            agente.observacao(obs)
            accao = agente.age()
            sucesso, recompensa = ambiente.agir(accao, agente)
            agente.avaliacaoEstadoAtual(recompensa)
            recompensa_total += recompensa
            passos += 1
            ambiente.atualizacao()
        
        epsilon = getattr(agente, 'epsilon', None)
        metricas.registar_episodio(
            recompensa_total,
            passos,
            ambiente.episodio_terminado,
            epsilon
        )
        agente.reset()
    
    return metricas


def executar_comparacao(ambiente_tipo: str, episodios: int, runs: int) -> Dict:
    """Executa múltiplas runs de Q-Learning e SARSA."""
    resultados = {
        'qlearning': [],
        'sarsa': []
    }
    
    for run in range(runs):
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{runs}")
        print(f"{'='*50}")
        
        # Q-Learning
        print("\nTreinando Q-Learning...")
        ambiente = criar_ambiente(ambiente_tipo)
        agente_q = AgenteQLearning(
            nome="QLearning",
            alpha=0.1,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        ambiente.adicionar_agente(agente_q)
        metricas_q = executar_treino(ambiente, agente_q, episodios)
        resultados['qlearning'].append(metricas_q)
        
        stats = metricas_q.estatisticas()
        print(f"  Recompensa média: {stats['recompensa_media']:.1f}")
        print(f"  Taxa sucesso: {stats['taxa_sucesso']*100:.1f}%")
        
        # SARSA
        print("\nTreinando SARSA...")
        ambiente = criar_ambiente(ambiente_tipo)
        agente_s = AgenteSARSA(
            nome="SARSA",
            alpha=0.1,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        ambiente.adicionar_agente(agente_s)
        metricas_s = executar_treino(ambiente, agente_s, episodios)
        resultados['sarsa'].append(metricas_s)
        
        stats = metricas_s.estatisticas()
        print(f"  Recompensa média: {stats['recompensa_media']:.1f}")
        print(f"  Taxa sucesso: {stats['taxa_sucesso']*100:.1f}%")
    
    return resultados


def plotar_comparacao(resultados: Dict, episodios: int, ambiente_tipo: str, caminho: str = None):
    """Gera gráficos comparativos lado-a-lado."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparação Q-Learning vs SARSA - Ambiente {ambiente_tipo.capitalize()}", fontsize=14)
    
    cores = {'qlearning': 'blue', 'sarsa': 'orange'}
    labels = {'qlearning': 'Q-Learning', 'sarsa': 'SARSA'}
    
    # Calcular médias e desvios padrão
    for algoritmo in ['qlearning', 'sarsa']:
        runs = resultados[algoritmo]
        
        # Recompensas
        todas_recompensas = np.array([m.recompensas for m in runs])
        media_recompensas = np.mean(todas_recompensas, axis=0)
        std_recompensas = np.std(todas_recompensas, axis=0)
        
        # Média móvel
        janela = 50
        media_movel = np.convolve(media_recompensas, np.ones(janela)/janela, mode='valid')
        
        # Plot recompensas
        ax1 = axes[0, 0]
        x = range(len(media_movel))
        ax1.plot(x, media_movel, color=cores[algoritmo], label=labels[algoritmo], linewidth=2)
        
        # Taxa de sucesso
        todas_sucessos = np.array([m.sucessos for m in runs], dtype=float)
        taxa_acumulada = np.cumsum(todas_sucessos, axis=1) / np.arange(1, episodios + 1)
        media_taxa = np.mean(taxa_acumulada, axis=0)
        
        ax2 = axes[0, 1]
        ax2.plot(media_taxa, color=cores[algoritmo], label=labels[algoritmo], linewidth=2)
        
        # Passos por episódio
        todos_passos = np.array([m.passos for m in runs])
        media_passos = np.mean(todos_passos, axis=0)
        media_movel_passos = np.convolve(media_passos, np.ones(janela)/janela, mode='valid')
        
        ax3 = axes[1, 0]
        ax3.plot(media_movel_passos, color=cores[algoritmo], label=labels[algoritmo], linewidth=2)
    
    # Configurar gráficos
    axes[0, 0].set_xlabel('Episódio')
    axes[0, 0].set_ylabel('Recompensa (média móvel)')
    axes[0, 0].set_title('Recompensa por Episódio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episódio')
    axes[0, 1].set_ylabel('Taxa de Sucesso')
    axes[0, 1].set_title('Taxa de Sucesso Acumulada')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Episódio')
    axes[1, 0].set_ylabel('Passos (média móvel)')
    axes[1, 0].set_title('Passos por Episódio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Estatísticas finais
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    texto = "ESTATÍSTICAS FINAIS\n" + "="*40 + "\n\n"
    for algoritmo in ['qlearning', 'sarsa']:
        runs = resultados[algoritmo]
        todas_stats = [m.estatisticas() for m in runs]
        
        media_rew = np.mean([s['ultimos_50_recompensa'] for s in todas_stats])
        std_rew = np.std([s['ultimos_50_recompensa'] for s in todas_stats])
        media_suc = np.mean([s['ultimos_50_sucesso'] for s in todas_stats]) * 100
        std_suc = np.std([s['ultimos_50_sucesso'] for s in todas_stats]) * 100
        media_passos = np.mean([s['passos_medios'] for s in todas_stats])
        
        texto += f"{labels[algoritmo]}:\n"
        texto += f"  Recompensa (últ. 50): {media_rew:.1f} ± {std_rew:.1f}\n"
        texto += f"  Sucesso (últ. 50): {media_suc:.1f}% ± {std_suc:.1f}%\n"
        texto += f"  Passos médios: {media_passos:.1f}\n\n"
    
    ax4.text(0.1, 0.9, texto, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if caminho:
        plt.savefig(caminho, dpi=150, bbox_inches='tight')
        print(f"\nGráfico guardado em: {caminho}")
    
    plt.show()


def exportar_csv(resultados: Dict, caminho: str):
    """Exporta resultados para CSV."""
    import csv
    
    with open(caminho, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algoritmo', 'run', 'episodio', 'recompensa', 'passos', 'sucesso'])
        
        for algoritmo, runs in resultados.items():
            for run_idx, metricas in enumerate(runs):
                for ep in range(len(metricas.recompensas)):
                    writer.writerow([
                        algoritmo,
                        run_idx + 1,
                        ep + 1,
                        metricas.recompensas[ep],
                        metricas.passos[ep],
                        int(metricas.sucessos[ep])
                    ])
    
    print(f"Dados exportados para: {caminho}")


def main():
    parser = argparse.ArgumentParser(description="Comparação Q-Learning vs SARSA")
    
    parser.add_argument('--ambiente', '-a', choices=['farol', 'labirinto'],
                        default='farol', help='Ambiente a usar')
    parser.add_argument('--episodios', '-e', type=int, default=500,
                        help='Episódios por run')
    parser.add_argument('--runs', '-r', type=int, default=3,
                        help='Número de runs')
    parser.add_argument('--output', '-o', type=str, default='resultados',
                        help='Diretório de output')
    
    args = parser.parse_args()
    
    # Criar diretório de output
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"COMPARAÇÃO Q-LEARNING vs SARSA")
    print(f"{'='*60}")
    print(f"Ambiente: {args.ambiente}")
    print(f"Episódios: {args.episodios}")
    print(f"Runs: {args.runs}")
    print(f"{'='*60}")
    
    # Executar comparação
    resultados = executar_comparacao(args.ambiente, args.episodios, args.runs)
    
    # Gerar gráfico
    grafico_path = output_dir / f"comparacao_{args.ambiente}.png"
    plotar_comparacao(resultados, args.episodios, args.ambiente, str(grafico_path))
    
    # Exportar CSV
    csv_path = output_dir / f"comparacao_{args.ambiente}.csv"
    exportar_csv(resultados, str(csv_path))
    
    print(f"\n{'='*60}")
    print("COMPARAÇÃO CONCLUÍDA!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
