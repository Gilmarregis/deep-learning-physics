# Deep Learning Aplicado à Física 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Um projeto demonstrando aplicações avançadas de **Deep Learning** na **Física**, implementando duas abordagens inovadoras:

1. **🔬 Resolução da Equação de Schrödinger** usando Physics-Informed Neural Networks (PINNs)
2. **⚡ Simulação de Dinâmica de Partículas** com aprendizado de máquina

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Características](#-características)
- [Instalação](#-instalação)
- [Uso Rápido](#-uso-rápido)
- [Arquitetura](#-arquitetura)
- [Teoria Física](#-teoria-física)
- [Resultados](#-resultados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

## 🎯 Visão Geral

Este projeto demonstra como **redes neurais** podem ser usadas para resolver problemas complexos em física, desde **mecânica quântica** até **simulações de sistemas dinâmicos**. Utilizamos técnicas de ponta como Physics-Informed Neural Networks (PINNs) que incorporam leis físicas diretamente no processo de aprendizado.

### 🔬 Aplicação 1: Mecânica Quântica
- **Objetivo**: Resolver a equação de Schrödinger para um oscilador harmônico
- **Método**: PINN (Physics-Informed Neural Network)
- **Resultado**: Função de onda ψ(x) do estado fundamental

### ⚡ Aplicação 2: Dinâmica Clássica
- **Objetivo**: Simular movimento de partículas sob forças físicas
- **Método**: Rede neural que aprende F = ma
- **Resultado**: Trajetórias realistas de partículas

## ✨ Características

### 🧠 Tecnologias de IA
- **TensorFlow 2.x** para deep learning
- **Diferenciação automática** para cálculo de derivadas
- **Otimização Adam** para treinamento eficiente
- **Arquiteturas personalizadas** para cada problema físico

### 🔬 Física Implementada
- **Equação de Schrödinger** independente do tempo
- **Oscilador harmônico quântico** (V = ½kx²)
- **Dinâmica newtoniana** (F = ma)
- **Forças gravitacionais e elásticas**

### 📊 Visualizações
- **Função de onda complexa** (partes real e imaginária)
- **Densidade de probabilidade quântica**
- **Trajetórias de partículas** em tempo real
- **Métricas de convergência** do treinamento

## 🛠 Instalação

### Pré-requisitos
- **Python 3.8+**
- **pip** (gerenciador de pacotes)
- **8GB+ RAM** (recomendado)
- **GPU** (opcional, mas acelera o treinamento)

### Passo a Passo

1. **Clone ou baixe o projeto**
```bash
# Se usando git:
git clone <url-do-repositorio>
cd deep-learning-physics

# Ou simplesmente baixe os arquivos para uma pasta
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Verifique a instalação**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado!')"
```

### 📦 Dependências

| Biblioteca | Versão | Propósito |
|------------|--------|----------|
| `tensorflow` | ≥2.10.0 | Deep learning e diferenciação automática |
| `numpy` | ≥1.21.0 | Computação numérica |
| `matplotlib` | ≥3.5.0 | Visualização de gráficos |
| `scipy` | ≥1.8.0 | Algoritmos científicos |
| `seaborn` | ≥0.11.0 | Visualizações estatísticas |

## 🚀 Uso Rápido

### Execução Simples
```bash
python run_physics_simulation.py
```

### Execução Personalizada
```python
from physics_deep_learning import QuantumNeuralNetwork, ParticlePhysicsSimulator

# Modelo quântico
quantum_model = QuantumNeuralNetwork(hidden_units=256)
# ... treinamento personalizado

# Simulador de partículas
particle_sim = ParticlePhysicsSimulator()
trajectory = particle_sim.simulate_trajectory([1.0, 2.0, 0.5, -0.3])
```

### Saídas Esperadas
- **Console**: Progresso do treinamento e métricas
- **Arquivo**: `physics_deep_learning_results.png` (visualizações)
- **Gráficos**: 4 subplots mostrando resultados completos

## 🏗 Arquitetura

### 🔬 QuantumNeuralNetwork