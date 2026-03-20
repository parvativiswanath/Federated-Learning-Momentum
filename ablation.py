"""
Ablation study for FedWAN.

Isolates the contribution of each component by running four variants:
  1. Vanilla FedAvg       -- plain SGD clients,  uniform aggregation
  2. FedAvg + NAG         -- NAG clients,         uniform aggregation
  3. FedAvg + KL weights  -- plain SGD clients,  KL-weighted aggregation
  4. FedWAN               -- NAG clients,         KL-weighted aggregation

Each variant is run on every (dataset, split) combination:
  Datasets : mnist, emnist_bymerge
  Splits   : proportional, dirichlet

Total runs: 4 variants × 2 datasets × 2 splits × 2 models = 32 runs.

Reuses fed_avg / fed_momentum_nag / fed_wan / get_label_distribution /
DATASET_CONFIGS from federated.py, and train / train_with_NAG / test from
model/train.py.

Output
------
- Console: per-round progress + final summary table
- ablation_results.csv: one row per (variant, dataset, split, model)
"""

import csv
import os
import time
from copy import deepcopy
from threading import Thread, Lock

import torch

from dataset import data_utils
from model.layers import CNN, DNN
from model.train import train, train_with_NAG, test

# Import aggregation helpers and config from federated.py
from federated import (
    fed_avg,
    fed_momentum_nag,
    fed_wan,
    get_label_distribution,
    client_data_sizes as fed_client_data_sizes,
    DATASET_CONFIGS,
    device,
    output_dir,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAINING_ROUNDS = 15
NUM_CLIENTS     = 5
DATASETS        = ['mnist', 'emnist_bymerge']
SPLITS          = ['proportional', 'dirichlet']
MODELS          = ['cnn', 'dnn']

VARIANTS = [
    # (display name,           use_nag, use_kl_weights)
    ("Vanilla FedAvg",         False,   False),
    ("FedAvg + NAG",           True,    False),
    ("FedAvg + KL weights",    False,   True),
    ("FedWAN",                 True,    True),
]

# ---------------------------------------------------------------------------
# Per-round client runner
# ---------------------------------------------------------------------------

def run_round(server_model, server_velocity, client_datasets, use_nag):
    """
    Train all clients in parallel for one round.

    Returns
    -------
    models        : list of trained client models
    velocities    : list of client velocities (empty list when use_nag=False)
    distributions : list of label-count dicts
    data_sizes    : list of dataset sizes per client
    """
    models, velocities, distributions, data_sizes = [], [], [], []
    lock = Lock()

    def client_fn(client_idx, _sm=server_model, _sv=server_velocity):
        loader    = data_utils.get_dataloader(client_datasets[client_idx])
        model     = deepcopy(_sm)
        dist      = get_label_distribution(loader)
        data_size = len(client_datasets[client_idx])

        if use_nag:
            vel = deepcopy(_sv)
            trained_model, trained_vel = train_with_NAG(model, loader, vel)
            with lock:
                models.append(trained_model)
                velocities.append(trained_vel)
                distributions.append(dist)
                data_sizes.append(data_size)
        else:
            trained_model = train(model, loader)
            with lock:
                models.append(trained_model)
                distributions.append(dist)
                data_sizes.append(data_size)

    threads = [Thread(target=client_fn, args=(c,)) for c in range(NUM_CLIENTS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return models, velocities, distributions, data_sizes


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg, model_type):
    if model_type == 'cnn':
        return CNN(
            in_channels=cfg['in_channels'],
            num_classes=cfg['num_classes'],
            input_size=cfg['input_size'],
        ).to(device)
    else:
        return DNN(
            in_channels=cfg['in_channels'],
            num_classes=cfg['num_classes'],
            input_size=cfg['input_size'],
        ).to(device)


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------

def checkpoint_path(name, dataset, split, model_type):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = name.lower().replace(" ", "_").replace("+", "plus")
    filename = f"ablation_{tag}_{dataset}_{split}_{model_type}.pt"
    return os.path.join(ckpt_dir, filename)


def run_variant(name, use_nag, use_kl_weights, dataset, split, model_type, train_set, test_set):
    """
    Run one FL variant for TRAINING_ROUNDS rounds on a given dataset and split.

    If a checkpoint already exists for this (name, dataset, split, model_type)
    combination, training is skipped and the saved metrics are returned directly.

    For "FedAvg + KL weights" (plain SGD + KL aggregation) we call fed_wan
    with dummy zero velocities and discard the returned velocity.

    Returns (final_accuracy, final_loss, total_time_s).
    """
    ckpt_path = checkpoint_path(name, dataset, split, model_type)

    print(f"\n{'='*65}", flush=True)
    print(f"  Variant : {name}", flush=True)
    print(f"  Dataset : {dataset}  |  Split: {split}  |  Model: {model_type.upper()}", flush=True)
    print(f"  NAG     : {use_nag}  |  KL weights: {use_kl_weights}", flush=True)
    print(f"{'='*65}", flush=True)

    # Resume from checkpoint if available
    if os.path.exists(ckpt_path):
        print(f"  [checkpoint found] Loading from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        return ckpt['final_acc'], ckpt['final_loss'], ckpt['total_time_s']

    cfg         = DATASET_CONFIGS[dataset]
    num_classes = cfg['num_classes']

    if split == 'dirichlet':
        client_datasets = data_utils.split_non_iid_dirichlet(
            train_set, NUM_CLIENTS, num_classes
        )
    else:
        client_datasets = data_utils.split_non_iid_class_proportional(
            train_set, NUM_CLIENTS, num_classes
        )

    test_loader     = data_utils.get_dataloader(test_set)
    server_model    = build_model(cfg, model_type)
    server_velocity = {n: torch.zeros_like(p) for n, p in server_model.named_parameters()}

    start_time = time.time()
    final_acc, final_loss = 0.0, 0.0

    for rnd in range(TRAINING_ROUNDS):
        models, velocities, distributions, data_sizes = run_round(
            server_model, server_velocity, client_datasets, use_nag
        )

        if use_nag and use_kl_weights:
            # FedWAN: NAG + KL-weighted aggregation
            fed_client_data_sizes.clear()
            fed_client_data_sizes.extend(data_sizes)
            server_model, server_velocity, _, _ = fed_wan(
                models, velocities, distributions, rnd, NUM_CLIENTS
            )

        elif use_nag:
            # FedAvg + NAG: NAG + uniform aggregation
            server_model, server_velocity, _ = fed_momentum_nag(models, velocities)

        elif use_kl_weights:
            # FedAvg + KL weights: plain SGD + KL-weighted aggregation.
            # fed_wan requires velocity lists — supply dummy zeros, discard result.
            fed_client_data_sizes.clear()
            fed_client_data_sizes.extend(data_sizes)
            dummy_velocities = [
                {n: torch.zeros_like(p) for n, p in server_model.named_parameters()}
                for _ in models
            ]
            server_model, _, _, _ = fed_wan(
                models, dummy_velocities, distributions, rnd, NUM_CLIENTS
            )

        else:
            # Vanilla FedAvg: plain SGD + uniform aggregation
            server_model = fed_avg(models)

        acc, loss = test(server_model, test_loader)
        elapsed   = time.time() - start_time
        print(
            f"  Round {rnd+1:2d}/{TRAINING_ROUNDS} | "
            f"Acc: {acc:.4f}  Loss: {loss:.4f}  Elapsed: {elapsed:.1f}s",
            flush=True,
        )
        final_acc, final_loss = acc, loss

    total_time = time.time() - start_time
    torch.save({
        'model_state_dict': server_model.state_dict(),
        'final_acc':        final_acc,
        'final_loss':       final_loss,
        'total_time_s':     total_time,
        'variant':          name,
        'dataset':          dataset,
        'split':            split,
        'model_type':       model_type,
    }, ckpt_path)
    print(f"  [checkpoint saved] {ckpt_path}", flush=True)
    return final_acc, final_loss, total_time


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []  # (variant, dataset, split, model_type, use_nag, use_kl, acc, loss, time)

    for dataset in DATASETS:
        cfg = DATASET_CONFIGS[dataset]
        print(f"\n{'#'*65}", flush=True)
        print(f"  Loading dataset: {dataset}", flush=True)
        print(f"{'#'*65}", flush=True)
        train_set = cfg['load_train']()
        test_set  = cfg['load_test']()

        for split in SPLITS:
            for model_type in MODELS:
                for name, use_nag, use_kl in VARIANTS:
                    acc, loss, t = run_variant(
                        name, use_nag, use_kl,
                        dataset, split, model_type,
                        train_set, test_set,
                    )
                    results.append((name, dataset, split, model_type, use_nag, use_kl, acc, loss, t))

    # ------------------------------------------------------------------
    # Save combined CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "fl_variant", "dataset", "split", "model", "nag", "agg_weights",
            "final_accuracy", "final_loss", "time_s",
        ])
        for name, dataset, split, model_type, use_nag, use_kl, acc, loss, t in results:
            writer.writerow([
                name, dataset, split, model_type.upper(),
                "Yes" if use_nag else "No",
                "KL-weighted" if use_kl else "Uniform",
                round(acc, 6),
                round(loss, 6),
                round(t, 2),
            ])
    print(f"\nResults saved to: {csv_path}", flush=True)

    # ------------------------------------------------------------------
    # Print summary table (grouped by dataset + split)
    # ------------------------------------------------------------------
    col_w   = [24, 5, 13, 11, 11, 10]
    headers = ["FL Variant", "NAG", "Agg Weights", "Final Acc", "Final Loss", "Time (s)"]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    sep = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    for dataset in DATASETS:
        for split in SPLITS:
            for model_type in MODELS:
                print(f"\n\n{'='*len(sep)}", flush=True)
                print(
                    f"  ABLATION RESULTS  |  dataset={dataset}  "
                    f"split={split}  model={model_type.upper()}  "
                    f"|  5 clients  |  15 rounds",
                    flush=True,
                )
                print(f"{'='*len(sep)}", flush=True)
                print(fmt_row(headers), flush=True)
                print(sep, flush=True)

                for name, ds, sp, mt, use_nag, use_kl, acc, loss, t in results:
                    if ds != dataset or sp != split or mt != model_type:
                        continue
                    print(fmt_row([
                        name,
                        "Yes" if use_nag else "No",
                        "KL-weighted" if use_kl else "Uniform",
                        f"{acc:.4f}",
                        f"{loss:.4f}",
                        f"{t:.1f}",
                    ]), flush=True)

                print(f"{'='*len(sep)}", flush=True)

    print("", flush=True)
