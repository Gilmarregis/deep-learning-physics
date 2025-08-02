#!/usr/bin/env python3
"""
Script principal para executar as simulações de física com deep learning
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from physics_deep_learning import visualize_results

def main():
    """
    Função principal para executar as simulações
    """
    try:
        print("Iniciando simulações de física com deep learning...")
        visualize_results()
        print("\nSimulações concluídas com sucesso!")
        
    except ImportError as e:
        print(f"Erro de importação: {e}")
        print("Instale as dependências com: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)