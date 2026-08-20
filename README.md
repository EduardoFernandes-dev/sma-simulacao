# Simulação Multiagente (SMA)

Simulação de sistemas multiagente em Python para estudar o comportamento de
agentes (aleatório, Q-Learning, SARSA) em dois ambientes (farol, labirinto), com
modos de treino, teste e visualização. Projeto académico com modelos, resultados
e testes.

## Stack

- Python
- numpy, matplotlib e pygame (visualização)

## Como correr

    python main.py                          # treino no ambiente farol com Q-Learning
    python main.py --ambiente labirinto --modo visual    # visualização (pygame)
    python main.py --agente sarsa --episodios 1000
    python comparar_algoritmos.py --ambiente farol --runs 3   # Q-Learning vs SARSA
    python -m pytest tests

Opções principais do `main.py`:

- `--ambiente`, `-a`: `farol` (default) ou `labirinto`
- `--modo`, `-m`: `treino` (default), `teste` ou `visual`
- `--agente`: `aleatorio`, `qlearning` (default) ou `sarsa`
- `--episodios`: número de episódios (default 500)
- `--config`: ficheiro de configuração em `configs/`

## Estrutura

- `src/`, núcleo da simulação (agentes, ambientes, sensores, visualização)
- `configs/`, `farol_config.json` e `labirinto_config.json`
- `modelos/`, modelos treinados (`*.pkl`)
- `resultados/`, figuras e resultados de execuções
- `tests/`, testes
- `requirements.txt`, dependências
