"""
Scaling experiments: FedWAN vs FedProx vs FedNova vs SCAFFOLD
on EMNIST ByMerge with varying client populations and partial participation.

Outputs
-------
- Per-run CSVs written by federated() to output_dir
- scaling_convergence_table.csv   — final/best accuracy + loss for every run
- fedwan_weights_*.png            — FedWAN per-round weight distributions (box plots)
"""

import os
import csv

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from federated import federated, output_dir

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
DATASET = 'emnist_bymerge'
ALGOS   = ['fedwan', 'fedprox', 'fednova', 'scaffold']
SPLIT   = 'dirichlet'
ALPHA   = 0.5

CONFIGS = [
    {'num_clients': 20,  'participation_rates': [1.0, 0.5, 0.2]},
    {'num_clients': 50,  'participation_rates': [1.0, 0.3, 0.1]},
    {'num_clients': 100, 'participation_rates': [1.0, 0.3, 0.1]},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_tag(num_clients, participation_rate):
    return f'_N{num_clients}_P{participation_rate}'


def _metrics_path(algo, num_clients, participation_rate):
    tag = _run_tag(num_clients, participation_rate)
    return os.path.join(output_dir, f'federated_metrics_{DATASET}_{algo}{tag}.csv')


def _weights_path(num_clients, participation_rate):
    tag = _run_tag(num_clients, participation_rate)
    return os.path.join(output_dir, f'fedwan_weights_{DATASET}{tag}.csv')


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all():
    for cfg in CONFIGS:
        n = cfg['num_clients']
        for p in cfg['participation_rates']:
            for algo in ALGOS:
                print(f"\n{'='*60}", flush=True)
                print(f"  {algo.upper()}  |  N={n}  P={p}", flush=True)
                print(f"{'='*60}", flush=True)
                federated(algo, DATASET,
                          num_clients=n,
                          participation_rate=p,
                          split=SPLIT,
                          alpha=ALPHA)


# ---------------------------------------------------------------------------
# Convergence table
# ---------------------------------------------------------------------------

def build_convergence_table():
    rows = []

    for cfg in CONFIGS:
        n = cfg['num_clients']
        for p in cfg['participation_rates']:
            for algo in ALGOS:
                path = _metrics_path(algo, n, p)
                if not os.path.exists(path):
                    print(f"  Missing: {path}", flush=True)
                    continue

                df = pd.read_csv(path)
                acc_col  = f'accuracy_{algo}'
                loss_col = f'loss_{algo}'

                final_acc  = df[acc_col].iloc[-1]
                final_loss = df[loss_col].iloc[-1]
                best_idx   = df[acc_col].idxmax()
                best_acc   = df[acc_col].iloc[best_idx]
                best_round = int(df['round'].iloc[best_idx])

                rows.append({
                    'algo':             algo,
                    'N':                n,
                    'participation':    p,
                    'final_acc':        round(final_acc,  4),
                    'final_loss':       round(final_loss, 4),
                    'best_acc':         round(best_acc,   4),
                    'best_round':       best_round,
                })

    table = pd.DataFrame(rows, columns=[
        'algo', 'N', 'participation', 'final_acc', 'final_loss', 'best_acc', 'best_round'
    ])

    out_path = os.path.join(output_dir, 'scaling_convergence_table.csv')
    table.to_csv(out_path, index=False)

    print(f"\nConvergence table saved to {out_path}\n", flush=True)
    print(table.to_string(index=False), flush=True)

    return table


# ---------------------------------------------------------------------------
# FedWAN weight distribution plots
# ---------------------------------------------------------------------------

def plot_weight_distributions():
    for cfg in CONFIGS:
        n = cfg['num_clients']
        rates = cfg['participation_rates']

        fig, axes = plt.subplots(1, len(rates), figsize=(6 * len(rates), 5), sharey=True)
        if len(rates) == 1:
            axes = [axes]

        for ax, p in zip(axes, rates):
            path = _weights_path(n, p)
            if not os.path.exists(path):
                ax.set_title(f'P={p} (no data)')
                continue

            df = pd.read_csv(path)
            weight_cols = [c for c in df.columns if c.startswith('w_')]
            rounds = df['round'].tolist()

            # One box per round: collect weight values across clients
            data_per_round = [df.loc[df['round'] == r, weight_cols].values.flatten()
                              for r in rounds]

            ax.boxplot(data_per_round, positions=rounds, widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor='#a8c8e8', alpha=0.7),
                       medianprops=dict(color='#1f77b4', linewidth=2))

            uniform = 1.0 / max(1, round(p * n))
            ax.axhline(uniform, color='red', linestyle='--', linewidth=1,
                       label=f'Uniform (1/{round(p*n)})')

            ax.set_xlabel('Round', fontsize=11)
            ax.set_title(f'P={p}  ({round(p*n)} clients/round)', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)

        axes[0].set_ylabel('Client Weight', fontsize=11)
        fig.suptitle(f'FedWAN Per-Round Weight Distribution  |  N={n}  |  EMNIST ByMerge',
                     fontsize=12, y=1.02)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'fedwan_weights_{DATASET}_N{n}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Weight plot saved to {out_path}", flush=True)
        plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all()
    build_convergence_table()
    plot_weight_distributions()
