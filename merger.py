import pandas as pd
import os


def merge(dataset='mnist'):
    algorithms = ['fedwan', 'fednag', 'mfl', 'mime', 'fedmom', 'fedavg']
    output_file = f'federated_metrics_{dataset}_combined.csv'

    metrics_df = pd.DataFrame()

    for algo in algorithms:
        input_file = f'federated_metrics_{dataset}_{algo}.csv'
        if os.path.exists(input_file):
            algo_df = pd.read_csv(input_file)
            metrics_df = pd.concat([metrics_df, algo_df], axis=1)
        else:
            print(f"File not found: {input_file}")

    metrics_df.to_csv(output_file, index=False)
    print(f"Combined metrics saved to {output_file}")
