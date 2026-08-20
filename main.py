#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulador import MotorDeSimulacao
from src.ambientes.farol import AmbienteFarol
from src.ambientes.labirinto import AmbienteLabirinto
from src.agentes.agente_aleatorio import AgenteAleatorio
from src.agentes.agente_qlearning import AgenteQLearning
from src.agentes.agente_sarsa import AgenteSARSA
from src.visualizacao.renderer import Renderer
from src.utils.metricas import MetricasSimulacao
from src.sensores import criar_sensor_farol_completo, criar_sensor_farol_limitado, criar_sensor_labirinto


def criar_argumentos():
    """Cria parser de argumentos da linha de comandos."""
    parser = argparse.ArgumentParser(
        description="Simulador de Sistemas Multi-Agente",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--ambiente', '-a',
        choices=['farol', 'labirinto'],
        default='farol',
        help='Ambiente a simular (default: farol)'
    )
    
    parser.add_argument(
        '--modo', '-m',
        choices=['treino', 'teste', 'visual'],
        default='treino',
        help='Modo de execução (default: treino)'
    )
    
    parser.add_argument(
        '--agente', '-g',
        choices=['aleatorio', 'qlearning', 'sarsa'],
        default='qlearning',
        help='Tipo de agente (default: qlearning)'
    )
    
    parser.add_argument(
        '--episodios', '-e',
        type=int,
        default=500,
        help='Número de episódios (default: 500)'
    )
    
    parser.add_argument(
        '--modelo',
        type=str,
        default=None,
        help='Caminho para carregar/guardar modelo'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Ficheiro de configuração JSON'
    )
    
    parser.add_argument(
        '--sem-grafico',
        action='store_true',
        help='Não mostrar gráfico no final'
    )
    
    parser.add_argument(
        '--auto-decay',
        action='store_true',
        default=True,
        help='Usar epsilon_decay automático baseado no número de episódios (default: True)'
    )
    
    parser.add_argument(
        '--no-auto-decay',
        action='store_true',
        help='Desativar epsilon_decay automático, usar valor fixo 0.995'
    )
    
    parser.add_argument(
        '--auto-alpha',
        action='store_true',
        default=False,
        help='Usar alpha decrescente durante o treino (começa alto, diminui)'
    )
    
    parser.add_argument(
        '--sensor', '-s',
        choices=['completo', 'limitado', 'nenhum'],
        default='limitado',
        help='Tipo de sensor a instalar no agente (default: limitado)'
    )
    
    return parser.parse_args()


def criar_ambiente(tipo: str, config: str = None) -> 'AmbienteFarol':
    """Cria o ambiente especificado."""
    if tipo == 'farol':
        if config:
            return AmbienteFarol.cria(config)
        else:
            # Configuração por defeito
            ambiente = AmbienteFarol(
                largura=10,
                altura=10,
                densidade_obstaculos=0.1
            )
            ambiente.objetivo = (9, 9)
            ambiente.posicao_inicial = (0, 0)
            ambiente._gerar_ambiente()
            return ambiente
    
    elif tipo == 'labirinto':
        if config:
            return AmbienteLabirinto.cria(config)
        else:
            # Configuração por defeito (grelha simples)
            ambiente = AmbienteLabirinto(
                largura=11,
                altura=11
            )
            ambiente._gerar_labirinto_dfs()
            return ambiente
    
    raise ValueError(f"Ambiente desconhecido: {tipo}")


def calcular_epsilon_decay(episodios: int, epsilon_inicio: float = 1.0, 
                            epsilon_min: float = 0.01, percentagem_convergencia: float = 0.8) -> float:
    episodio_alvo = int(episodios * percentagem_convergencia)
    if episodio_alvo <= 0:
        episodio_alvo = 1
    
    decay = (epsilon_min / epsilon_inicio) ** (1.0 / episodio_alvo)
    return decay


def calcular_alpha_decay(episodios: int, alpha_inicio: float = 0.3, 
                          alpha_min: float = 0.05, percentagem_convergencia: float = 0.7) -> float:

    episodio_alvo = int(episodios * percentagem_convergencia)
    if episodio_alvo <= 0:
        episodio_alvo = 1
    
    decay = (alpha_min / alpha_inicio) ** (1.0 / episodio_alvo)
    return decay


def criar_agente(tipo: str, modo_teste: bool = False, episodios: int = 500, 
                 auto_decay: bool = True, auto_alpha: bool = False) -> 'Agente':
    """Cria o agente especificado com epsilon_decay e alpha otimizados."""
    if tipo == 'aleatorio':
        return AgenteAleatorio(nome="AgAleatorio")
    
    # Calcular ou usar epsilon_decay fixo
    if auto_decay:
        epsilon_decay = calcular_epsilon_decay(episodios)
        print(f"  epsilon_decay AUTOMÁTICO: {epsilon_decay:.6f}")
    else:
        epsilon_decay = 0.995
        print(f"  epsilon_decay FIXO: {epsilon_decay}")
    
    # Calcular ou usar alpha fixo
    if auto_alpha:
        alpha_inicio = 0.3
        alpha_min = 0.05
        alpha_decay = calcular_alpha_decay(episodios, alpha_inicio, alpha_min)
        print(f"  alpha ADAPTATIVO: {alpha_inicio} → {alpha_min} (decay={alpha_decay:.6f})")
    else:
        alpha_inicio = 0.1
        alpha_decay = 1.0  # Sem decay
        alpha_min = 0.01
        print(f"  alpha FIXO: {alpha_inicio}")
    
    if tipo == 'qlearning':
        agente = AgenteQLearning(
            nome="AgQLearning",
            alpha=alpha_inicio,
            gamma=0.95,
            epsilon=1.0 if not modo_teste else 0.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=0.01,
            alpha_decay=alpha_decay,
            alpha_min=alpha_min
        )
        agente.modo_aprendizagem = not modo_teste
        return agente
    
    elif tipo == 'sarsa':
        agente = AgenteSARSA(
            nome="AgSARSA",
            alpha=0.1,
            gamma=0.95,
            epsilon=1.0 if not modo_teste else 0.0,
            epsilon_decay=epsilon_decay,
            epsilon_min=0.01
        )
        agente.modo_aprendizagem = not modo_teste
        return agente
    
    raise ValueError(f"Agente desconhecido: {tipo}")


def executar_treino(ambiente, agente, episodios: int, metricas: MetricasSimulacao):
    """Executa simulação em modo treino."""
    from src.core.observacao import Observacao
    
    usa_sensores = len(agente.sensores) > 0
    
    print(f"\n{'='*50}")
    print(f"MODO TREINO - {episodios} episódios")
    print(f"Ambiente: {ambiente}")
    print(f"Agente: {agente}")
    if usa_sensores:
        print(f"Perceção: Via SENSORES ({agente.sensores[0].nome})")
    else:
        print(f"Perceção: Via Ambiente (observacaoPara)")
    print(f"{'='*50}\n")
    
    for episodio in range(episodios):
        ambiente.reset()
        recompensa_total = 0
        passos = 0
        
        while not ambiente.episodio_terminado and passos < 200:
            # Observação - usar sensores se instalados
            if usa_sensores:
                dados_sensor = agente.perceber()
                obs = Observacao.from_sensor_data(dados_sensor)
            else:
                obs = ambiente.observacaoPara(agente)
            
            agente.observacao(obs)
            
            # Ação
            accao = agente.age()
            
            # Executar
            sucesso, recompensa = ambiente.agir(accao, agente)
            agente.avaliacaoEstadoAtual(recompensa)
            
            recompensa_total += recompensa
            passos += 1
            ambiente.atualizacao()
        
        # Registar métricas
        epsilon = getattr(agente, 'epsilon', None)
        metricas.registar_episodio(
            recompensa_total, 
            passos, 
            ambiente.episodio_terminado,
            epsilon
        )
        
        # Progresso
        if (episodio + 1) % 50 == 0:
            stats = metricas.estatisticas()
            print(f"Ep {episodio+1:4d} | Rew: {stats['ultimos_50_recompensa']:7.1f} | "
                  f"Sucesso: {stats['ultimos_50_sucesso']*100:5.1f}% | "
                  f"ε: {epsilon:.3f}" if epsilon else "")
    
    agente.reset()
    return metricas


def executar_teste(ambiente, agente, num_testes: int = 10):
    """
    Executa simulação em modo teste/avaliação.
    
    Conforme enunciado (Secção C.2 - Modo de Teste):
    - Política do agente FIXA (não modifica Q-table)
    - Avalia: taxa de sucesso, passos médios, recompensa média, recompensa descontada
    """
    from src.core.observacao import Observacao
    
    usa_sensores = len(agente.sensores) > 0
    gamma = getattr(agente, 'gamma', 0.95)
    
    # GARANTIR que a política é FIXA (modo teste = sem aprendizagem)
    agente.modo_aprendizagem = False
    agente.epsilon = 0.0  # Sem exploração aleatória
    
    print(f"\n{'='*60}")
    print(f"MODO TESTE - Avaliação com Política Fixa")
    print(f"{'='*60}")
    print(f"Ambiente: {ambiente}")
    print(f"Agente: {agente}")
    print(f"Episódios de teste: {num_testes}")
    if hasattr(agente, 'q_table'):
        print(f"Estados na Q-Table: {len(agente.q_table)}")
    if usa_sensores:
        print(f"Perceção: Via SENSORES ({agente.sensores[0].nome})")
    print(f"{'='*60}\n")
    
    # Verificar se Q-table está carregada
    if hasattr(agente, 'q_table') and len(agente.q_table) == 0:
        print("⚠️  AVISO: Q-Table vazia! Use --modelo para carregar um modelo treinado.")
        print("   Exemplo: python main.py --modo teste --modelo modelo_treinado.pkl\n")
    
    # Métricas de avaliação
    resultados = []
    
    for teste in range(num_testes):
        ambiente.reset()
        recompensa_total = 0
        recompensa_descontada = 0
        passos = 0
        
        while not ambiente.episodio_terminado and passos < 200:
            # Observação
            if usa_sensores:
                dados_sensor = agente.perceber()
                obs = Observacao.from_sensor_data(dados_sensor)
            else:
                obs = ambiente.observacaoPara(agente)
            
            agente.observacao(obs)
            
            # Ação com política FIXA (epsilon=0, sem aprendizagem)
            accao = agente.age()
            
            # Executar
            _, recompensa = ambiente.agir(accao, agente)
            
            # Acumular métricas
            recompensa_total += recompensa
            recompensa_descontada += (gamma ** passos) * recompensa
            passos += 1
            ambiente.atualizacao()
        
        sucesso = ambiente.episodio_terminado
        resultados.append({
            'sucesso': sucesso,
            'passos': passos,
            'recompensa': recompensa_total,
            'recompensa_descontada': recompensa_descontada
        })
        
        status = "✓" if sucesso else "✗"
        print(f"  Ep {teste+1:2d}: {status} | Passos: {passos:3d} | Rew: {recompensa_total:7.1f} | Rew(γ): {recompensa_descontada:7.1f}")
    
    # Calcular métricas finais
    sucessos = sum(1 for r in resultados if r['sucesso'])
    taxa_sucesso = 100 * sucessos / num_testes
    passos_medio = sum(r['passos'] for r in resultados) / num_testes
    recompensa_media = sum(r['recompensa'] for r in resultados) / num_testes
    recompensa_desc_media = sum(r['recompensa_descontada'] for r in resultados) / num_testes
    
    # Apenas para episódios com sucesso
    if sucessos > 0:
        passos_medio_sucesso = sum(r['passos'] for r in resultados if r['sucesso']) / sucessos
    else:
        passos_medio_sucesso = float('inf')
    
    print(f"\n{'='*60}")
    print(f"MÉTRICAS DE AVALIAÇÃO (Modo Teste)")
    print(f"{'='*60}")
    print(f"  Taxa de Sucesso:       {taxa_sucesso:.1f}% ({sucessos}/{num_testes})")
    print(f"  Passos Médios (total): {passos_medio:.1f}")
    print(f"  Passos Médios (suc):   {passos_medio_sucesso:.1f}")
    print(f"  Recompensa Média:      {recompensa_media:.2f}")
    print(f"  Recompensa Descontada: {recompensa_desc_media:.2f}")
    print(f"{'='*60}")
    
    return resultados, {
        'taxa_sucesso': taxa_sucesso,
        'passos_medio': passos_medio,
        'recompensa_media': recompensa_media,
        'recompensa_desc_media': recompensa_desc_media
    }


def executar_visual(ambiente, agente, episodios: int = 10):
    """Executa simulação com visualização."""
    from src.core.observacao import Observacao
    
    usa_sensores = len(agente.sensores) > 0
    
    # GARANTIR política FIXA no modo visual (igual ao modo teste)
    agente.modo_aprendizagem = False
    agente.epsilon = 0.0  # Sem exploração aleatória
    
    print("\n" + "="*50)
    print("MODO VISUAL - Política Fixa (ε=0)")
    print("Controlos: ESPAÇO=Pause, I=Info, ESC=Sair")
    if usa_sensores:
        print(f"Perceção: Via SENSORES ({agente.sensores[0].nome})")
    else:
        print("Perceção: Via Ambiente (observacaoPara)")
    print("="*50 + "\n")
    
    renderer = Renderer(largura=900, altura=600)
    renderer.inicializar()
    renderer.configurar(ambiente, None)
    
    try:
        for episodio in range(episodios):
            ambiente.reset()
            
            while not ambiente.episodio_terminado and ambiente.passos_episodio < 200:
                # Processar eventos
                if not renderer.processar_eventos():
                    return
                
                # Observação - usar sensores se instalados (IGUAL ao treino/teste!)
                if usa_sensores:
                    dados_sensor = agente.perceber()
                    obs = Observacao.from_sensor_data(dados_sensor)
                else:
                    obs = ambiente.observacaoPara(agente)
                agente.observacao(obs)
                
                # Ação
                accao = agente.age()
                
                # Executar
                sucesso, recompensa = ambiente.agir(accao, agente)
                agente.avaliacaoEstadoAtual(recompensa)
                ambiente.atualizacao()
                
                # Renderizar
                renderer.renderizar()
                
                # Delay para visualização
                import time
                time.sleep(0.1)
            
            print(f"Episódio {episodio+1} - Sucesso: {ambiente.episodio_terminado}, "
                  f"Passos: {ambiente.passos_episodio}, "
                  f"Recompensa: {agente.recompensa_acumulada:.1f}")
            
            agente.reset()
    
    finally:
        renderer.fechar()


def main():
    """Função principal."""
    import random
    import numpy as np
    
    args = criar_argumentos()
    
    # Variável para guardar a seed
    seed_ambiente = None
    modelo_info = {}
    
    # Se for modo teste ou visual, carregar modelo primeiro para obter seed
    if args.modo in ('teste', 'visual') and args.modelo and Path(args.modelo).exists():
        # Carregar seed do modelo antes de criar ambiente
        import pickle
        with open(args.modelo, 'rb') as f:
            dados = pickle.load(f)
        if isinstance(dados, dict) and 'seed' in dados:
            seed_ambiente = dados.get('seed')
            if seed_ambiente:
                print(f"\n  Usando seed do treino: {seed_ambiente}")
    
    # Se for modo treino, gerar nova seed
    if args.modo == 'treino':
        seed_ambiente = random.randint(1, 100000)
        print(f"\n  Seed do ambiente: {seed_ambiente}")
    
    # Aplicar seed se existir
    if seed_ambiente:
        random.seed(seed_ambiente)
        np.random.seed(seed_ambiente)
    
    # Criar ambiente (agora com seed aplicada)
    config_path = args.config or f"configs/{args.ambiente}_config.json"
    if not Path(config_path).exists():
        config_path = None
    ambiente = criar_ambiente(args.ambiente, config_path)
    
    # Criar agente (com epsilon_decay e alpha otimizados)
    modo_teste = args.modo == 'teste'
    episodios = args.episodios
    auto_decay = not args.no_auto_decay
    auto_alpha = args.auto_alpha
    agente = criar_agente(args.agente, modo_teste, episodios, auto_decay, auto_alpha)
    
    # Instalar sensor se especificado
    if args.sensor != 'nenhum':
        if args.ambiente == 'farol':
            if args.sensor == 'completo':
                sensor = criar_sensor_farol_completo()
            else:  # limitado
                sensor = criar_sensor_farol_limitado()
        else:  # labirinto
            sensor = criar_sensor_labirinto()
        
        agente.instala(sensor)
        sensor.configurar(ambiente)
        print(f"  Sensor instalado: {sensor.nome}")
    
    # Adicionar agente ao ambiente
    ambiente.adicionar_agente(agente)
    
    # Carregar modelo se especificado (Q-table)
    if args.modelo and Path(args.modelo).exists() and hasattr(agente, 'carregar_modelo'):
        modelo_info = agente.carregar_modelo(args.modelo)
    
    # Executar
    if args.modo == 'visual':
        executar_visual(ambiente, agente, episodios=20)
    
    elif args.modo == 'teste':
        # Modo teste - requer modelo carregado
        if not args.modelo or not Path(args.modelo).exists():
            print("\n⚠️  MODO TESTE requer um modelo treinado!")
            print("   Primeiro treine e guarde um modelo:")
            print("   python main.py --modo treino --episodios 500 --modelo modelo.pkl")
            print("\n   Depois teste o modelo:")
            print("   python main.py --modo teste --modelo modelo.pkl")
            return
        
        resultados, metricas_teste = executar_teste(ambiente, agente, num_testes=args.episodios if args.episodios != 500 else 10)
        
        # Gráfico de resultados do teste
        if not args.sem_grafico and len(resultados) > 1:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            
            eps = list(range(1, len(resultados) + 1))
            rews = [r['recompensa'] for r in resultados]
            passos = [r['passos'] for r in resultados]
            sucs = [1 if r['sucesso'] else 0 for r in resultados]
            
            # Recompensa por episódio
            axes[0].bar(eps, rews, color=['green' if s else 'red' for s in sucs], alpha=0.7)
            axes[0].axhline(y=metricas_teste['recompensa_media'], color='blue', linestyle='--', label=f"Média: {metricas_teste['recompensa_media']:.1f}")
            axes[0].set_xlabel('Episódio de Teste')
            axes[0].set_ylabel('Recompensa')
            axes[0].set_title('Recompensa por Episódio')
            axes[0].legend()
            
            # Passos por episódio
            axes[1].bar(eps, passos, color=['green' if s else 'red' for s in sucs], alpha=0.7)
            axes[1].axhline(y=metricas_teste['passos_medio'], color='blue', linestyle='--', label=f"Média: {metricas_teste['passos_medio']:.1f}")
            axes[1].set_xlabel('Episódio de Teste')
            axes[1].set_ylabel('Passos')
            axes[1].set_title('Passos por Episódio')
            axes[1].legend()
            
            # Métricas resumo
            axes[2].axis('off')
            texto = f"""MÉTRICAS DE TESTE
            
Taxa de Sucesso: {metricas_teste['taxa_sucesso']:.1f}%
Passos Médios: {metricas_teste['passos_medio']:.1f}
Recompensa Média: {metricas_teste['recompensa_media']:.2f}
Rew. Descontada: {metricas_teste['recompensa_desc_media']:.2f}

Verde = Sucesso
Vermelho = Falhou"""
            axes[2].text(0.1, 0.5, texto, fontsize=12, verticalalignment='center', family='monospace')
            
            plt.suptitle(f'Resultados Modo Teste - {args.ambiente}', fontsize=14)
            plt.tight_layout()
            
            caminho = f"resultados/{args.ambiente}_teste.png"
            plt.savefig(caminho, dpi=150)
            print(f"\nGráfico guardado em: {caminho}")
            plt.show()
    
    else:  # treino
        metricas = MetricasSimulacao()
        metricas.nome_experiencia = f"{args.ambiente}_{args.agente}"
        
        executar_treino(ambiente, agente, args.episodios, metricas)
        
        # Guardar modelo (com seed do ambiente)
        if args.modelo and hasattr(agente, 'guardar_modelo'):
            agente.guardar_modelo(args.modelo, seed=seed_ambiente)
        
        # Resumo
        metricas.imprimir_resumo()
        
        # Gráfico
        if not args.sem_grafico:
            metricas.plotar_curva_aprendizagem(
                caminho=f"resultados/{metricas.nome_experiencia}_curva.png",
                mostrar=True
            )


if __name__ == "__main__":
    main()
