import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for Colab
import matplotlib.pyplot as plt

from federated import federated, output_dir

DATASET = 'emnist_bymerge'
ALGOS   = ['fedwan', 'fedprox', 'fednova', 'scaffold']

COLORS = {
    'fedwan':   '#1f77b4',
    'fedprox':  '#ff7f0e',
    'fednova':  '#2ca02c',
    'scaffold': '#d62728',
}
LABELS = {
    'fedwan':   'FedWAN',
    'fedprox':  'FedProx',
    'fednova':  'FedNova',
    'scaffold': 'SCAFFOLD',
}


def run_all():
    for algo in ALGOS:
        print(f"\n{'='*50}", flush=True)
        print(f"  {algo.upper()} on {DATASET}", flush=True)
        print(f"{'='*50}", flush=True)
        federated(algo, DATASET)


def plot():
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))

    for algo in ALGOS:
        csv_path = os.path.join(output_dir, f'federated_metrics_{DATASET}_{algo}_N5_P1.0.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found — skipping {algo}", flush=True)
            continue
        df = pd.read_csv(csv_path)
        rounds = df['round']
        ax_acc.plot(rounds,  df[f'accuracy_{algo}'], label=LABELS[algo],
                    color=COLORS[algo], linewidth=2, marker='o', markersize=3)
        ax_loss.plot(rounds, df[f'loss_{algo}'],     label=LABELS[algo],
                     color=COLORS[algo], linewidth=2, marker='o', markersize=3)

    for ax, ylabel, title in [
        (ax_acc,  'Accuracy', 'Test Accuracy vs Round'),
        (ax_loss, 'Loss',     'Test Loss vs Round'),
    ]:
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title}\n(EMNIST ByMerge, Non-IID)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'compare_noniid_emnist.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


if __name__ == "__main__":
    run_all()
    plot()
