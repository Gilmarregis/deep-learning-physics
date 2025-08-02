"""
Deep Learning Aplicado à Física
===============================

Este módulo implementa duas aplicações principais de deep learning na física:
1. Resolução da equação de Schrödinger usando redes neurais (Physics-Informed Neural Networks)
2. Simulação de dinâmica de partículas com aprendizado de máquina

Autor: Sistema de IA
Data: 2024
Versão: 1.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns

# Configuração para reprodutibilidade dos resultados
# Isso garante que os experimentos possam ser replicados
np.random.seed(42)
tf.random.set_seed(42)

class QuantumNeuralNetwork:
    """
    Rede Neural Informada pela Física (PINN) para resolver a equação de Schrödinger 1D
    
    Esta classe implementa uma rede neural que aprende a resolver a equação de Schrödinger
    para um oscilador harmônico quântico. A rede é treinada usando a própria equação
    diferencial como função de perda, sem necessidade de dados de treinamento tradicionais.
    
    Atributos:
        domain_size (int): Tamanho do domínio espacial para discretização
        hidden_units (int): Número de neurônios nas camadas ocultas
        model (keras.Sequential): Modelo da rede neural
    """
    
    def __init__(self, domain_size=100, hidden_units=128):
        """
        Inicializa a rede neural quântica
        
        Args:
            domain_size (int): Número de pontos no domínio espacial (padrão: 100)
            hidden_units (int): Neurônios por camada oculta (padrão: 128)
        """
        self.domain_size = domain_size
        self.hidden_units = hidden_units
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Constrói a arquitetura da rede neural para aproximar a função de onda
        
        A rede tem 4 camadas ocultas com ativação tanh (adequada para funções suaves)
        e uma camada de saída com 2 neurônios (parte real e imaginária da função de onda)
        
        Returns:
            keras.Sequential: Modelo da rede neural construído
        """
        model = keras.Sequential([
            # Camada de entrada: recebe posição x
            layers.Dense(self.hidden_units, activation='tanh', input_shape=(1,)),
            
            # Camadas ocultas: aproximam a função de onda complexa
            layers.Dense(self.hidden_units, activation='tanh'),
            layers.Dense(self.hidden_units, activation='tanh'),
            layers.Dense(self.hidden_units, activation='tanh'),
            
            # Camada de saída: parte real e imaginária da função de onda ψ(x)
            layers.Dense(2)  # [ψ_real(x), ψ_imag(x)]
        ])
        return model
    
    def schrodinger_loss(self, x, psi_pred):
        """
        Calcula a função de perda baseada na equação de Schrödinger
        
        A equação de Schrödinger independente do tempo é:
        -ℏ²/2m * d²ψ/dx² + V(x)*ψ = E*ψ
        
        Para o oscilador harmônico: V(x) = ½kx² = ½x² (assumindo k=1)
        
        Args:
            x (tf.Tensor): Pontos espaciais onde avaliar a equação
            psi_pred: Não utilizado (mantido para compatibilidade)
            
        Returns:
            tf.Tensor: Valor da função de perda (erro na equação de Schrödinger)
        """
        # Diferenciação automática aninhada para calcular derivadas de segunda ordem
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)  # Observa x para calcular d²ψ/dx²
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)  # Observa x para calcular dψ/dx
                
                # Predição da rede neural: ψ(x) = ψ_real(x) + i*ψ_imag(x)
                psi = self.model(x)
                psi_real = psi[:, 0:1]  # Parte real
                psi_imag = psi[:, 1:2]  # Parte imaginária
            
            # Primeira derivada: dψ/dx
            dpsi_real_dx = tape1.gradient(psi_real, x)
            dpsi_imag_dx = tape1.gradient(psi_imag, x)
            
        # Segunda derivada: d²ψ/dx²
        d2psi_real_dx2 = tape2.gradient(dpsi_real_dx, x)
        d2psi_imag_dx2 = tape2.gradient(dpsi_imag_dx, x)
        
        # Potencial do oscilador harmônico: V(x) = ½x²
        V = 0.5 * x**2
        
        # Energia do estado fundamental do oscilador harmônico
        # Para ℏ = m = ω = 1, temos E₀ = ½ℏω = 0.5
        E = 0.5
        
        # Equação de Schrödinger para partes real e imaginária
        # Parte real: -½d²ψ_real/dx² + V*ψ_real = E*ψ_real
        schrodinger_real = -0.5 * d2psi_real_dx2 + V * psi_real - E * psi_real
        
        # Parte imaginária: -½d²ψ_imag/dx² + V*ψ_imag = E*ψ_imag
        schrodinger_imag = -0.5 * d2psi_imag_dx2 + V * psi_imag - E * psi_imag
        
        # Retorna o erro quadrático médio da equação
        return tf.reduce_mean(tf.square(schrodinger_real) + tf.square(schrodinger_imag))
    
    def boundary_loss(self, x_boundary):
        """
        Implementa as condições de contorno: ψ(±∞) = 0
        
        Para o oscilador harmônico, a função de onda deve tender a zero
        nas extremidades do domínio (aproximando ±∞)
        
        Args:
            x_boundary (tf.Tensor): Pontos nas extremidades do domínio
            
        Returns:
            tf.Tensor: Penalidade por violação das condições de contorno
        """
        psi_boundary = self.model(x_boundary)
        # Penaliza qualquer valor não-zero nas extremidades
        return tf.reduce_mean(tf.square(psi_boundary))
    
    def normalization_loss(self, x):
        """
        Garante a normalização da função de onda: ∫|ψ(x)|²dx = 1
        
        A densidade de probabilidade |ψ(x)|² = ψ_real² + ψ_imag² deve
        integrar para 1 em todo o domínio espacial.
        
        Args:
            x (tf.Tensor): Pontos do domínio para integração numérica
            
        Returns:
            tf.Tensor: Penalidade por desvio da normalização
        """
        psi = self.model(x)
        psi_real = psi[:, 0:1]
        psi_imag = psi[:, 1:2]
        
        # Densidade de probabilidade: |ψ(x)|²
        probability_density = psi_real**2 + psi_imag**2
        
        # Integração numérica usando a regra do trapézio
        dx = x[1] - x[0]  # Espaçamento entre pontos
        integral = tf.reduce_sum(probability_density) * dx
        
        # Penaliza desvio da normalização (integral deve ser 1)
        return tf.square(integral - 1.0)
    
    def train_step(self, x_train, x_boundary):
        """
        Executa um passo de treinamento da rede neural
        
        Combina três tipos de perda:
        1. Equação de Schrödinger (física)
        2. Condições de contorno
        3. Normalização da função de onda
        
        Args:
            x_train (tf.Tensor): Pontos internos do domínio
            x_boundary (tf.Tensor): Pontos nas extremidades
            
        Returns:
            tuple: (perda_total, perda_schrodinger, perda_contorno, perda_normalização)
        """
        with tf.GradientTape() as tape:
            # Calcula todas as componentes da perda
            loss_schrodinger = self.schrodinger_loss(x_train, None)
            loss_boundary = self.boundary_loss(x_boundary)
            loss_norm = self.normalization_loss(x_train)
            
            # Combina as perdas com pesos diferentes
            # Peso maior para condições de contorno (10x) para garantir convergência
            total_loss = loss_schrodinger + 10.0 * loss_boundary + loss_norm
        
        # Calcula gradientes e atualiza parâmetros
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, loss_schrodinger, loss_boundary, loss_norm

class ParticlePhysicsSimulator:
    """
    Simulador de física de partículas usando deep learning
    
    Esta classe implementa um simulador que aprende as leis da física
    (gravidade + força elástica) a partir de dados e pode prever
    trajetórias de partículas em tempo real.
    
    Atributos:
        model (keras.Sequential): Rede neural que aprende F = ma
    """
    
    def __init__(self):
        """
        Inicializa o simulador de partículas
        
        Constrói uma rede neural que mapeia estados de partículas
        (posição e velocidade) para acelerações.
        """
        self.model = self._build_dynamics_model()
    
    def _build_dynamics_model(self):
        """
        Constrói o modelo de rede neural para aprender dinâmica de partículas
        
        A rede mapeia: [x, y, vx, vy] → [ax, ay]
        Onde (x,y) é posição, (vx,vy) é velocidade, (ax,ay) é aceleração
        
        Returns:
            keras.Sequential: Modelo para predição de acelerações
        """
        model = keras.Sequential([
            # Entrada: estado da partícula [x, y, vx, vy]
            layers.Dense(64, activation='relu', input_shape=(4,)),
            
            # Camadas ocultas: aprendem as leis da física
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            
            # Saída: aceleração [ax, ay] baseada em F = ma
            layers.Dense(2)
        ])
        return model
    
    def generate_training_data(self, n_samples=10000):
        """
        Gera dados sintéticos de treinamento para dinâmica de partículas
        
        Simula um sistema com:
        - Força gravitacional: Fg = mg (direção -y)
        - Força elástica: Fe = -kx (oscilador harmônico)
        
        Args:
            n_samples (int): Número de amostras de treinamento
            
        Returns:
            tuple: (estados, acelerações) para treinamento
        """
        # Gera estados iniciais aleatórios: [x, y, vx, vy]
        states = np.random.uniform(-5, 5, (n_samples, 4))
        
        # Array para armazenar acelerações correspondentes
        accelerations = np.zeros((n_samples, 2))
        
        # Calcula aceleração para cada estado usando leis da física
        for i in range(n_samples):
            x, y, vx, vy = states[i]
            
            # Constantes físicas
            g = -9.81  # Aceleração gravitacional (m/s²)
            k = 1.0    # Constante elástica (N/m)
            
            # Segunda lei de Newton: F = ma → a = F/m (assumindo m=1)
            ax = -k * x      # Força elástica horizontal
            ay = g - k * y   # Gravidade + força elástica vertical
            
            accelerations[i] = [ax, ay]
        
        return states, accelerations
    
    def train_dynamics(self, epochs=1000):
        """
        Treina a rede neural para aprender dinâmica de partículas
        
        A rede aprende a mapear estados (posição, velocidade) para
        acelerações, efetivamente aprendendo F = ma.
        
        Args:
            epochs (int): Número de épocas de treinamento
            
        Returns:
            keras.callbacks.History: Histórico do treinamento
        """
        # Gera dados de treinamento
        X, y = self.generate_training_data()
        
        # Configura o modelo para treinamento
        self.model.compile(
            optimizer='adam',           # Otimizador adaptativo
            loss='mse',                # Erro quadrático médio
            metrics=['mae']            # Erro absoluto médio para monitoramento
        )
        
        # Treina o modelo
        history = self.model.fit(
            X, y,                      # Dados de entrada e saída
            epochs=epochs,             # Número de épocas
            batch_size=32,             # Tamanho do lote
            validation_split=0.2,      # 20% dos dados para validação
            verbose=0                  # Treinamento silencioso
        )
        
        return history
    
    def simulate_trajectory(self, initial_state, time_steps=1000, dt=0.01):
        """
        Simula a trajetória de uma partícula usando a rede neural treinada
        
        Usa integração numérica de Euler para propagar o estado da partícula
        no tempo, com acelerações preditas pela rede neural.
        
        Args:
            initial_state (list): Estado inicial [x₀, y₀, vx₀, vy₀]
            time_steps (int): Número de passos de tempo para simular
            dt (float): Passo de tempo (segundos)
            
        Returns:
            np.ndarray: Trajetória completa [N×4] com estados em cada tempo
        """
        trajectory = [initial_state]  # Lista para armazenar trajetória
        current_state = initial_state.copy()  # Estado atual (mutável)
        
        # Simula evolução temporal
        for _ in range(time_steps):
            # Prediz aceleração usando a rede neural treinada
            acceleration = self.model.predict(
                current_state.reshape(1, -1), 
                verbose=0
            )[0]
            
            # Integração de Euler: método numérico simples
            # v(t+dt) = v(t) + a(t)*dt
            current_state[2] += acceleration[0] * dt  # vx += ax*dt
            current_state[3] += acceleration[1] * dt  # vy += ay*dt
            
            # x(t+dt) = x(t) + v(t)*dt
            current_state[0] += current_state[2] * dt  # x += vx*dt
            current_state[1] += current_state[3] * dt  # y += vy*dt
            
            # Armazena o novo estado
            trajectory.append(current_state.copy())
        
        return np.array(trajectory)

def visualize_results():
    """
    Função principal para visualizar os resultados de ambos os modelos
    
    Executa o treinamento completo e gera visualizações mostrando:
    1. Função de onda quântica resolvida
    2. Convergência do treinamento quântico
    3. Trajetórias de partículas simuladas
    4. Métricas de performance dos modelos
    """
    # Configura estilo dos gráficos
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ==========================================
    # PARTE 1: MODELO QUÂNTICO
    # ==========================================
    print("Treinando modelo quântico...")
    quantum_model = QuantumNeuralNetwork()
    
    # Define domínio espacial para o oscilador harmônico
    x_train = tf.constant(np.linspace(-3, 3, 100).reshape(-1, 1), dtype=tf.float32)
    x_boundary = tf.constant([[-3.0], [3.0]], dtype=tf.float32)  # Extremidades
    
    # Configura otimizador global
    global optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Loop de treinamento personalizado
    losses = []
    for epoch in range(2000):
        total_loss, loss_sch, loss_bound, loss_norm = quantum_model.train_step(
            x_train, x_boundary
        )
        
        # Monitora progresso a cada 200 épocas
        if epoch % 200 == 0:
            print(f"Época {epoch}: Loss total = {total_loss:.6f}")
        
        losses.append(total_loss.numpy())
    
    # Gera pontos para plotagem da função de onda
    x_plot = np.linspace(-3, 3, 200)
    psi_pred = quantum_model.model(x_plot.reshape(-1, 1)).numpy()
    
    # Subplot 1: Função de onda quântica
    axes[0, 0].plot(x_plot, psi_pred[:, 0], label='Parte Real', linewidth=2)
    axes[0, 0].plot(x_plot, psi_pred[:, 1], label='Parte Imaginária', linewidth=2)
    axes[0, 0].plot(x_plot, psi_pred[:, 0]**2 + psi_pred[:, 1]**2, 
                    label='Densidade de Probabilidade', linewidth=2, linestyle='--')
    axes[0, 0].set_title('Função de Onda Quântica (Oscilador Harmônico)')
    axes[0, 0].set_xlabel('Posição (x)')
    axes[0, 0].set_ylabel('ψ(x)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Convergência do treinamento quântico
    axes[0, 1].plot(losses)
    axes[0, 1].set_title('Convergência do Modelo Quântico')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_yscale('log')  # Escala logarítmica para melhor visualização
    axes[0, 1].grid(True, alpha=0.3)
    
    # ==========================================
    # PARTE 2: SIMULADOR DE PARTÍCULAS
    # ==========================================
    print("Treinando simulador de partículas...")
    particle_sim = ParticlePhysicsSimulator()
    history = particle_sim.train_dynamics(epochs=500)
    
    # Define estados iniciais para diferentes partículas
    initial_states = [
        [2.0, 3.0, 1.0, 0.0],    # Partícula 1: posição (2,3), velocidade (1,0)
        [-1.5, 2.0, 0.5, -1.0], # Partícula 2: posição (-1.5,2), velocidade (0.5,-1)
        [0.0, 4.0, -1.0, 0.0]   # Partícula 3: posição (0,4), velocidade (-1,0)
    ]
    
    # Cores para distinguir trajetórias
    colors = ['red', 'blue', 'green']
    
    # Simula e plota trajetórias
    for i, (initial_state, color) in enumerate(zip(initial_states, colors)):
        trajectory = particle_sim.simulate_trajectory(initial_state, time_steps=500)
        
        # Plota trajetória
        axes[1, 0].plot(trajectory[:, 0], trajectory[:, 1], 
                       color=color, label=f'Partícula {i+1}', linewidth=2)
        
        # Marca posição inicial
        axes[1, 0].scatter(initial_state[0], initial_state[1], 
                          color=color, s=100, marker='o', edgecolor='black')
    
    # Subplot 3: Trajetórias de partículas
    axes[1, 0].set_title('Trajetórias de Partículas (Deep Learning)')
    axes[1, 0].set_xlabel('Posição X')
    axes[1, 0].set_ylabel('Posição Y')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')  # Proporção 1:1 para visualização realista
    
    # Subplot 4: Histórico de treinamento do simulador
    axes[1, 1].plot(history.history['loss'], label='Loss de Treinamento')
    axes[1, 1].plot(history.history['val_loss'], label='Loss de Validação')
    axes[1, 1].set_title('Treinamento do Simulador de Partículas')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Mean Squared Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Salva e exibe os resultados
    plt.tight_layout()
    plt.savefig('physics_deep_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Relatório final
    print("\n=== Resultados ===")
    print(f"Modelo quântico treinado com loss final: {losses[-1]:.6f}")
    print(f"Simulador de partículas - Loss final: {history.history['loss'][-1]:.6f}")
    print("Gráficos salvos como 'physics_deep_learning_results.png'")

# Ponto de entrada principal
if __name__ == "__main__":
    print("=== Deep Learning Aplicado à Física ===")
    print("Este programa demonstra duas aplicações:")
    print("1. Resolução da equação de Schrödinger usando redes neurais")
    print("2. Simulação de dinâmica de partículas com deep learning\n")
    
    # Verifica configuração do sistema
    print(f"TensorFlow versão: {tf.__version__}")
    print(f"GPU disponível: {tf.config.list_physical_devices('GPU')}\n")
    
    # Executa as simulações
    visualize_results()