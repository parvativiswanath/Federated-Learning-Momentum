import pandas as pd
import os

def merge():
    # Define the algorithms and corresponding files
    algorithms = ['fedwan', 'fednag', 'mfl', 'mime', 'fedmom']
    output_file = 'federated_metrics_new.csv'

    # Create an empty DataFrame for the combined metrics file
    columns = [
        'time_fedwan', 'accuracy_fedwan', 'loss_fedwan',
        'time_fednag', 'accuracy_fednag', 'loss_fednag',
        'time_mfl', 'accuracy_mfl', 'loss_mfl',
        'time_mime', 'accuracy_mime', 'loss_mime',
        'time_fedmom', 'accuracy_fedmom', 'loss_fedmom'
    ]
    metrics_df = pd.DataFrame(columns=columns)

    # Process each algorithm's CSV file
    for algo in algorithms:
        input_file = f'federated_metrics_{algo}_non_iid_lr0.001.csv'
        if os.path.exists(input_file):
            # Read the algorithm-specific file
            algo_df = pd.read_csv(input_file)
            # Add prefix to the column names based on the algorithm
            algo_df.columns = [f"{col}" for col in algo_df.columns]
            # Merge data into the main DataFrame
            metrics_df = pd.concat([metrics_df, algo_df], axis=1)
        else:
            print(f"File not found: {input_file}")

    # Save the combined metrics to the output file
    metrics_df.to_csv(output_file, index=False)
    print(f"Combined metrics saved to {output_file}")
