from collections import Counter
from copy import deepcopy
from threading import Thread, Lock

import logging
import os
import torch
import time
import csv

logger = logging.getLogger('fedwan')


def _configure_logger(log_path):
    """Point the shared 'fedwan' logger at a new per-run log file."""
    _logger = logging.getLogger('fedwan')
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(threadName)s] %(message)s'))
    _logger.addHandler(fh)
    _logger.propagate = False

from dataset import data_utils
from model.layers import CNN, DNN, LogisticRegression
from model.train import (test, train, train_with_momentum, train_with_NAG, train_mime,
                         train_fedprox, train_fednova, train_scaffold)
from merger import merge

client_models = []
client_velocities = []
client_distributions = []
client_data_sizes = []
client_grads = []
client_steps = []            # FedNova: local gradient step counts (tau_i)
client_scaffold_deltas = []  # SCAFFOLD: list of (client_idx, delta_c_i) tuples

client_models_lock = Lock()
momentum = 0.9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

_drive_path = '/content/drive/MyDrive/fedwan_results'
if os.path.isdir('/content/drive/MyDrive'):
    os.makedirs(_drive_path, exist_ok=True)
    output_dir = _drive_path
    print(f"Google Drive detected — outputs will be saved to {output_dir}", flush=True)
else:
    output_dir = '.'

# Dataset configurations: each entry defines the loaders, model class,
# and input geometry so no hardcoding is needed anywhere else.
DATASET_CONFIGS = {
    'mnist': {
        'load_train': lambda: data_utils.load_mnist_dataset(isTrainDataset=True),
        'load_test':  lambda: data_utils.load_mnist_dataset(isTrainDataset=False),
        'model':      'cnn',
        'in_channels': 1,
        'input_size':  28,
        'num_classes': 10,
    },
    'cifar10': {
        'load_train': lambda: data_utils.load_cifar10_dataset(isTrainDataset=True),
        'load_test':  lambda: data_utils.load_cifar10_dataset(isTrainDataset=False),
        'model':      'dnn',
        'in_channels': 3,
        'input_size':  32,
        'num_classes': 10,
    },
    'emnist_bymerge': {
        'load_train': lambda: data_utils.load_emnist_bymerge_dataset(isTrainDataset=True),
        'load_test':  lambda: data_utils.load_emnist_bymerge_dataset(isTrainDataset=False),
        'model':      'cnn',
        'in_channels': 1,
        'input_size':  28,
        'num_classes': 47,
    },
    'svhn': {
        'load_train': lambda: data_utils.load_svhn_dataset(isTrainDataset=True),
        'load_test':  lambda: data_utils.load_svhn_dataset(isTrainDataset=False),
        'model':      'dnn',
        'in_channels': 3,
        'input_size':  32,
        'num_classes': 10,
    },
}



def get_label_distribution(data_loader):
        label_counts = Counter()

        # Count label occurrences
        for _, labels in data_loader:
            label_counts.update(labels.tolist())

        # Normalize to get the distribution
        distribution = {label: count for label, count in label_counts.items()}
        return distribution

def calculate_kl_divergence(client_distribution, global_distribution):
        kl_divergence = 0.0
        for label, count in client_distribution.items():
            p = torch.tensor(count)
            q = global_distribution[label]
            kl_divergence += p * torch.log(p / q)
        return kl_divergence



def client_training(serverModel, clientDatasets, client, round):
    global client_models
    client_train_set = clientDatasets[client]
    trainLoader = data_utils.get_dataloader(client_train_set)
    client_model = deepcopy(serverModel)
    trained_client_model = train(client_model, trainLoader)
    client_models_lock.acquire()
    client_models.append(trained_client_model)
    client_models_lock.release()

    logger.debug(f"Client {client+1} done")

def client_training_momentum(serverModel, serverVelocity, clientDatasets, client, round, nesterov):
    global client_models
    global client_velocities
    global client_distributions
    global client_data_sizes
    client_train_set = clientDatasets[client]
    client_data_size = len(clientDatasets[client])
    trainLoader = data_utils.get_dataloader(client_train_set)
    client_model = deepcopy(serverModel)
    client_velocity = deepcopy(serverVelocity)
    if nesterov:
        trained_client_model, trained_client_velocity = train_with_NAG(client_model, trainLoader, client_velocity)
    else:
        trained_client_model, trained_client_velocity = train_with_momentum(client_model, trainLoader, client_velocity)
    client_distribution = get_label_distribution(trainLoader)
    client_models_lock.acquire()
    client_models.append(trained_client_model)
    client_velocities.append(trained_client_velocity)
    client_distributions.append(client_distribution)
    client_data_sizes.append(client_data_size)
    client_models_lock.release()

    logger.debug(f"Client {client+1} done")

def client_training_mime(serverModel, serverVelocity, clientDatasets, client, round):
    global client_models
    global client_data_sizes
    global client_grads
    client_train_set = clientDatasets[client]
    client_data_size = len(clientDatasets[client])
    trainLoader = data_utils.get_dataloader(client_train_set)
    client_model = deepcopy(serverModel)

    trained_client_model, trained_client_gradients = train_mime(client_model, trainLoader, serverVelocity)

    client_models_lock.acquire()
    client_models.append(trained_client_model)
    client_data_sizes.append(client_data_size)
    client_grads.append(trained_client_gradients)
    client_models_lock.release()

    logger.debug(f"Client {client+1} done")


def client_training_fedprox(serverModel, clientDatasets, client, round):
    global client_models, client_data_sizes
    client_train_set = clientDatasets[client]
    trainLoader = data_utils.get_dataloader(client_train_set)
    client_model = deepcopy(serverModel)
    global_params = {name: param.clone().detach() for name, param in serverModel.named_parameters()}
    trained_client_model = train_fedprox(client_model, trainLoader, global_params)
    client_models_lock.acquire()
    client_models.append(trained_client_model)
    client_data_sizes.append(len(client_train_set))
    client_models_lock.release()
    logger.debug(f"Client {client+1} done")


def client_training_fednova(serverModel, clientDatasets, client, round):
    global client_models, client_data_sizes, client_steps
    client_train_set = clientDatasets[client]
    trainLoader = data_utils.get_dataloader(client_train_set)
    client_model = deepcopy(serverModel)
    trained_client_model, tau = train_fednova(client_model, trainLoader)
    client_models_lock.acquire()
    client_models.append(trained_client_model)
    client_data_sizes.append(len(client_train_set))
    client_steps.append(tau)
    client_models_lock.release()
    logger.debug(f"Client {client+1} done")


def client_training_scaffold(serverModel, server_c, per_client_c, clientDatasets, client, round):
    global client_models, client_scaffold_deltas
    client_train_set = clientDatasets[client]
    trainLoader = data_utils.get_dataloader(client_train_set)
    client_model = deepcopy(serverModel)
    trained_client_model, delta_c_i = train_scaffold(client_model, trainLoader, per_client_c[client], server_c)
    client_models_lock.acquire()
    client_models.append(trained_client_model)
    client_scaffold_deltas.append((client, delta_c_i))
    client_models_lock.release()
    logger.debug(f"Client {client+1} done")


def fed_avg(client_models):
    logger.debug(f'NO OF CLIENT MODELS RECEIVED: {len(client_models)}')
    averaged_model = deepcopy(client_models[0])
    with torch.no_grad():
        for model in client_models[1:]:
            for param1, param2 in zip(averaged_model.parameters(), model.parameters()):
                param1.data += param2.data
        for param in averaged_model.parameters():
            param.data /= len(client_models)
    return averaged_model


def fed_momentum_nag(client_models,client_velocities):
    averaged_model = deepcopy(client_models[0])
    averaged_velocity = deepcopy(client_velocities[0])
    with torch.no_grad():
        # Average the models
        for model in client_models[1:]:
            for param1, param2 in zip(averaged_model.parameters(), model.parameters()):
                param1.data += param2.data
        for param in averaged_model.parameters():
            param.data /= len(client_models)
        # Average the velocities
        vel_start = time.perf_counter()
        for velocity in client_velocities[1:]:
            for name in averaged_velocity:
                averaged_velocity[name] += velocity[name]
        for name in averaged_velocity:
            averaged_velocity[name] /= len(client_velocities)
        vel_agg_time = time.perf_counter() - vel_start
    return averaged_model, averaged_velocity, vel_agg_time

def fed_wan(client_models, client_velocities, client_distributions, round, num_clients):
    averaged_velocity = deepcopy(client_velocities[0])
    averaged_model = deepcopy(client_models[0])

    global client_data_sizes
    logger.debug(f'client_data_sizes: {client_data_sizes}')
    logger.debug(f'CLIENT DISTRIBUTIONS: {client_distributions}')
    total_size = sum(client_data_sizes)

    # Build global distribution by summing actual client label counts
    global_distribution = {}
    for distribution in client_distributions:
        for label, count in distribution.items():
            if label not in global_distribution:
                global_distribution[label] = 0
            global_distribution[label] += count

    logger.debug(f'GLOBAL DISTRIBUTION: {global_distribution}')
    # Normalize the distribution
    for label in global_distribution:
        global_distribution[label] /= total_size

    # Initialise weights
    model_device = next(client_models[0].parameters()).device
    weights = torch.zeros(len(client_distributions), dtype=torch.float, device=model_device)

    # Calculate weights
    for client in range(len(client_distributions)):
        kli = calculate_kl_divergence(client_distributions[client], global_distribution)
        logger.debug(f"client {client} kli: {kli}")
        weights[client] = 1/kli
    logger.debug(f'weights before normalizing: {weights}')
    total_weight = weights.sum()
    for client in range(len(client_distributions)):
        weights[client] = weights[client] / total_weight
    logger.debug(f'weights for aggregation: {weights}')

    with torch.no_grad():
        # Initialize model parameters to zero
        for param in averaged_model.parameters():
            param.data.zero_()

        # Weighted aggregation of models
        for model, weight in zip(client_models, weights):
            for param_avg, param_client in zip(averaged_model.parameters(), model.parameters()):
                param_avg.data += weight * param_client.data

    # Average the velocities
    vel_start = time.perf_counter()
    for velocity in client_velocities[1:]:
        for name in averaged_velocity:
            averaged_velocity[name] += velocity[name]
    for name in averaged_velocity:
        averaged_velocity[name] /= len(client_velocities)
    vel_agg_time = time.perf_counter() - vel_start

    return averaged_model, averaged_velocity, vel_agg_time


def fed_mom(client_models, serverModel, globalVelocity):
    global_params = serverModel.state_dict()
    weighted_diff = {key: torch.zeros_like(tensor) for key, tensor in global_params.items()}
    global_velocity = {key: torch.zeros_like(tensor) for key, tensor in global_params.items()}

    for client_model in client_models:
        client_state = client_model.state_dict()
        for key in global_params.keys():
            weighted_diff[key] += (global_params[key] - client_state[key])

    for key in global_params.keys():
        global_velocity[key] = momentum * global_velocity[key] - 0.1 * weighted_diff[key]
        global_params[key] = global_params[key] + global_velocity[key]

    serverModel.load_state_dict(global_params)

    return serverModel, global_velocity


def fed_nova(client_models, client_data_sizes, client_steps, serverModel):
    total_data = sum(client_data_sizes)
    p = [n / total_data for n in client_data_sizes]
    tau_eff = sum(pi * tau for pi, tau in zip(p, client_steps))
    server_state = serverModel.state_dict()
    weighted_grad_sum = {key: torch.zeros_like(v) for key, v in server_state.items()}
    with torch.no_grad():
        for model, pi, tau in zip(client_models, p, client_steps):
            client_state = model.state_dict()
            for key in server_state:
                g_i = (server_state[key] - client_state[key]) / tau
                weighted_grad_sum[key] += pi * g_i
        new_state = {key: server_state[key] - tau_eff * weighted_grad_sum[key]
                     for key in server_state}
    new_model = deepcopy(serverModel)
    new_model.load_state_dict(new_state)
    return new_model


def scaffold_aggregate(client_models, client_scaffold_deltas, server_c, num_clients):
    averaged_model = deepcopy(client_models[0])
    with torch.no_grad():
        for model in client_models[1:]:
            for p1, p2 in zip(averaged_model.parameters(), model.parameters()):
                p1.data += p2.data
        for p in averaged_model.parameters():
            p.data /= len(client_models)
    # Update server control variate: c += (1/N) * sum(delta_c_i)
    with torch.no_grad():
        for name in server_c:
            avg_delta = sum(dc[name] for _, dc in client_scaffold_deltas) / num_clients
            server_c[name] = server_c[name] + avg_delta
    return averaged_model, server_c


def mime(client_models, client_grads, serverVelocity):
    logger.debug(f'mime: {len(client_models)} models, {len(client_grads[0])} grad keys')
    for i, client_grad in enumerate(client_grads):
        logger.debug(f"Keys in client_grads[{i}]: {list(client_grad.keys())}")
    averaged_model = deepcopy(client_models[0])
    with torch.no_grad():
        for model in client_models[1:]:
            for param1, param2 in zip(averaged_model.parameters(), model.parameters()):
                param1.data += param2.data
        for param in averaged_model.parameters():
            param.data /= len(client_models)

    # Update server velocity from averaged client gradients
    global_dict = averaged_model.state_dict()
    avg_clients_grads = deepcopy(global_dict)
    vel_start = time.perf_counter()
    for key in global_dict.keys():
        if key not in client_grads[0]:
            logger.debug(f"Skipping key {key} as it is not found in client_grads.")
            continue
        avg_clients_grads[key] = torch.stack([client_grads[i][key].float() for i in range(len(client_grads))], 0).mean(0)
        serverVelocity[key] = (1-momentum)*avg_clients_grads[key] + momentum*serverVelocity[key]
    vel_agg_time = time.perf_counter() - vel_start

    return averaged_model, serverVelocity, vel_agg_time

def print_velocities(velocity, label="Velocity"):
    lines = [f"{label}:"]
    for name, tensor in velocity.items():
        lines.append(f" - {name}: shape={tensor.shape}, size={tensor.numel()}")
    logger.debug("\n".join(lines))

def federated(algo, dataset='mnist'):
    start_time = time.time()
    num_clients = 5
    training_rounds = 15

    log_path = os.path.join(output_dir, f'federated_log_{dataset}_{algo}.log')
    _configure_logger(log_path)

    cfg = DATASET_CONFIGS[dataset]
    trainSet = cfg['load_train']()
    testSet  = cfg['load_test']()
    num_classes = cfg['num_classes']
    in_channels = cfg['in_channels']
    input_size  = cfg['input_size']

    clientDatasets = data_utils.split_non_iid_class_proportional(trainSet, num_clients, num_classes)
    testLoader = data_utils.get_dataloader(testSet)

    filename = os.path.join(output_dir, f'federated_metrics_{dataset}_{algo}.csv')
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["round", f"time_{algo}", f"accuracy_{algo}", f"loss_{algo}", "agg_time_s"])

    global client_models
    global client_velocities
    global client_distributions
    global client_grads
    global client_steps
    global client_scaffold_deltas

    if cfg['model'] == 'cnn':
        serverModel = CNN(in_channels=in_channels, num_classes=num_classes, input_size=input_size).to(device)
    elif cfg['model'] == 'dnn':
        serverModel = DNN(in_channels=in_channels, num_classes=num_classes, input_size=input_size).to(device)
    else:
        serverModel = LogisticRegression(in_channels=in_channels, num_classes=num_classes, input_size=input_size).to(device)

    serverVelocity = {name: torch.zeros_like(param) for name, param in serverModel.named_parameters()}

    # SCAFFOLD-specific persistent state (per-client and server control variates)
    if algo == "scaffold":
        per_client_c = [
            {name: torch.zeros_like(param) for name, param in serverModel.named_parameters()}
            for _ in range(num_clients)
        ]
        server_c = {name: torch.zeros_like(param) for name, param in serverModel.named_parameters()}

    for rnd in range(training_rounds):
        print(f"[{algo}/{dataset}] Round {rnd+1}/{training_rounds}", flush=True)

        client_models.clear()
        client_velocities.clear()
        client_distributions.clear()
        client_data_sizes.clear()
        client_grads.clear()
        client_steps.clear()
        client_scaffold_deltas.clear()
        client_threads = []
        for client in range(num_clients):
            if algo == "fedavg":
                t = Thread(
                    target=client_training,
                    args=(serverModel, clientDatasets, client, rnd)
                )
            elif algo == "mfl":
                t = Thread(
                    target=client_training_momentum,
                    args=(serverModel, serverVelocity, clientDatasets, client, rnd, False)
                )
            elif algo == "fednag":
                t = Thread(
                    target=client_training_momentum,
                    args=(serverModel, serverVelocity, clientDatasets, client, rnd, True)
                )
            elif algo == "fedwan":
                t = Thread(
                    target=client_training_momentum,
                    args=(serverModel, serverVelocity, clientDatasets, client, rnd, True)
                )
            elif algo == "fedmom":
                t = Thread(
                    target=client_training,
                    args=(serverModel, clientDatasets, client, rnd)
                )
            elif algo == "mime":
                t = Thread(
                    target=client_training_mime,
                    args=(serverModel, serverVelocity, clientDatasets, client, rnd)
                )
            elif algo == "fedprox":
                t = Thread(
                    target=client_training_fedprox,
                    args=(serverModel, clientDatasets, client, rnd)
                )
            elif algo == "fednova":
                t = Thread(
                    target=client_training_fednova,
                    args=(serverModel, clientDatasets, client, rnd)
                )
            elif algo == "scaffold":
                t = Thread(
                    target=client_training_scaffold,
                    args=(serverModel, server_c, per_client_c, clientDatasets, client, rnd)
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            t.start()
            client_threads.append(t)

        for t in client_threads:
            t.join()

        agg_start = time.perf_counter()
        if algo == "fedavg":
            serverModel = fed_avg(client_models)
        elif algo == "mfl" or algo == "fednag":
            serverModel, serverVelocity, _ = fed_momentum_nag(client_models, client_velocities)
        elif algo == "fedwan":
            serverModel, serverVelocity, _ = fed_wan(client_models, client_velocities, client_distributions, rnd, num_clients)
        elif algo == "fedmom":
            serverModel, serverVelocity = fed_mom(client_models, serverModel, serverVelocity)
        elif algo == "mime":
            serverModel, serverVelocity, _ = mime(client_models, client_grads, serverVelocity)
        elif algo == "fedprox":
            serverModel = fed_avg(client_models)
        elif algo == "fednova":
            serverModel = fed_nova(client_models, client_data_sizes, client_steps, serverModel)
        elif algo == "scaffold":
            serverModel, server_c = scaffold_aggregate(client_models, client_scaffold_deltas, server_c, num_clients)
            # Update each client's local control variate
            for client_idx, delta_c in client_scaffold_deltas:
                for name in per_client_c[client_idx]:
                    per_client_c[client_idx][name] = per_client_c[client_idx][name] + delta_c[name]
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        agg_time = time.perf_counter() - agg_start

        testAcc, testLoss = test(serverModel, testLoader)
        curr_time = time.time() - start_time
        print(f"  -> Acc: {testAcc:.4f}  Loss: {testLoss:.4f}", flush=True)
        logger.debug(f"Round {rnd+1} done | acc={testAcc:.4f} loss={testLoss:.4f} agg={agg_time:.4f}s")

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rnd + 1, curr_time, testAcc, testLoss, round(agg_time, 6)])


if __name__ == "__main__":
    dataset = 'mnist'  # options: 'mnist', 'cifar10', 'emnist_bymerge', 'svhn'
    for algo in ['fedwan', 'fednag', 'mfl', 'mime', 'fedmom', 'fedavg']:
        federated(algo, dataset)
    merge(dataset)
