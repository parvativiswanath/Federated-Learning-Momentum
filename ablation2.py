"""
Ablation study — Session 2: EMNIST (bymerge)
Run in parallel with ablation1.py (MNIST) on a separate Colab session.
"""

from ablation import run_ablation

if __name__ == "__main__":
    run_ablation(['emnist_bymerge'], "ablation_results_emnist.csv")
