"""
Ablation study — Session 1: MNIST
Run in parallel with ablation2.py (EMNIST) on a separate Colab session.
"""

from ablation import run_ablation

if __name__ == "__main__":
    run_ablation(['mnist'], "ablation_results_mnist.csv")
