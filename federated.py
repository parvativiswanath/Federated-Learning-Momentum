from collections import Counter
from copy import deepcopy
from threading import Thread, Lock

import os
import torch
import time
import csv

from dataset import data_utils
from model.layers import CNN, DNN, LogisticRegression
from model.train import test, train, train_with_momentum, train_with_NAG, train_mime
from merger import merge

client_models = []
client_velocities = []
client_distributions = []
client_data_sizes = []
client_grads = []

client_models_lock = Lock()
momentum = 0.9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

_drive_path = '/content/drive/MyDrive/fedwan_results'
if os.path.isdir('/content/drive/MyDrive'):
    os.makedirs(_drive_path, exist_ok=True)
    output_dir = _drive_path
    print(f"Google Drive detected — outputs will be saved to {output_dir}")
else:
    output_dir = '.'


def model_bytes(model):
    return sum(p.nelement() * p.element_size() for p in model.parameters())


def velocity_bytes(velocity):
    return sum(v.nelement() * v.element_size() for v in velocity.values())


def comm_bytes_per_round(algo, model, velocity, num_clients):
    """Returns (uplink_per_client, downlink_per_client, total) in bytes.
    fedavg/fedmom: model only. mfl/fednag/fedwan/mime: model + velocity."""
    m = model_bytes(model)
    v = velocity_bytes(velocity)
    if algo in ("fedavg", "fedmom"):
        uplink, downlink = m, m
    else:  # mfl, fednag, fedwan, mime all transfer velocity
        uplink, downlink = m + v, m + v
    return uplink, downlink, num_clients * (uplink + downlink)

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
    #client_train_set = clientDatasets[client][round]
    client_train_set = clientDatasets[client]
    trainLoader = data_utils.get_dataloader(client_train_set)
    client_model = deepcopy(serverModel)
    trained_client_model = train(client_model, trainLoader)
    client_models_lock.acquire()
    client_models.append(trained_client_model)
    client_models_lock.release()

    print(f"Client {client+1} done")

def client_training_momentum(serverModel, serverVelocity, clientDatasets, client, round, nesterov):
    global client_models
    global client_velocities
    global client_distributions
    global client_data_sizes
    #client_train_set = clientDatasets[client][round]
    client_train_set = clientDatasets[client]
    client_data_size = len(clientDatasets[client])
    #client_data_size = len(clientDatasets[client][round])
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

    print(f"Client {client+1} done")

def client_training_mime(serverModel, serverVelocity, clientDatasets, client, round):
    global client_models
    global client_data_sizes
    global client_grads
    #client_train_set = clientDatasets[client][round]
    client_train_set = clientDatasets[client]
    client_data_size = len(clientDatasets[client])
    #client_data_size = len(clientDatasets[client][round])
    trainLoader = data_utils.get_dataloader(client_train_set)
    client_model = deepcopy(serverModel)
    
    trained_client_model, trained_client_gradients = train_mime(client_model, trainLoader, serverVelocity)

    client_models_lock.acquire()
    client_models.append(trained_client_model)
    client_data_sizes.append(client_data_size)
    client_grads.append(trained_client_gradients)
    client_models_lock.release()

    print(f"Client {client+1} done")


def fed_avg(client_models):
    print('NO OF CLIENT MODELS RECEIVED: ', len(client_models))
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

def fed_wan(client_models,client_velocities, client_distributions, round, num_clients):
    averaged_velocity = deepcopy(client_velocities[0])
    averaged_model = deepcopy(client_models[0])

    global client_data_sizes
    print('client_data_sizes: ',client_data_sizes)
    print('CLIENT DISTRIBUTIONS:')
    print(client_distributions)
    total_size = sum(client_data_sizes)
    global_distribution = {}
    for distribution in client_distributions:
        for label, count in distribution.items():
            if label not in global_distribution:
                global_distribution[label] = 800
            #global_distribution[label] += count

    
    print('GLOBAL DISTRIBUTION:')
    print(global_distribution)
    # Normalize the distribution
    for label, count in global_distribution.items():
        global_distribution[label] /= total_size
    
    #Initialise weights
    model_device = next(client_models[0].parameters()).device
    weights = torch.zeros(len(client_distributions), dtype=torch.float, device=model_device)

    # Calculate weights
    for client in range(len(client_distributions)):
        print("client: ",client)
        print("client_distributions: ",client_distributions[client])
        kli = calculate_kl_divergence(client_distributions[client], global_distribution)
        print("kli: ",kli)
        weights[client] = 1/kli
    print('weights before normalizing: ', weights)
    total_weight = weights.sum()
    for client in range(len(client_distributions)):
        weights[client] = weights[client] / total_weight
    print('weights for aggregation: ', weights)

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
        # global_velocity[key] = global_params[key] - weighted_diff[key]
        # global_params[key] = global_velocity[key] + momentum * (global_velocity[key] - globalVelocity[key])
        global_velocity[key] = momentum * global_velocity[key] - 0.1 * weighted_diff[key]
        global_params[key] = global_params[key] + global_velocity[key]

    serverModel.load_state_dict(global_params)
    
    return serverModel, global_velocity


def mime(client_models, client_grads, serverVelocity):
    print('len:',len(client_models))
    print('len:', len(client_grads[0]))
    for i, client_grad in enumerate(client_grads):
        print(f"Keys in client_grads[{i}]: {client_grad.keys()}")
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
            print(f"Skipping key {key} as it is not found in client_grads.")
            continue
        avg_clients_grads[key] = torch.stack([client_grads[i][key].float() for i in range(len(client_grads))], 0).mean(0)
        serverVelocity[key] = (1-momentum)*avg_clients_grads[key] + momentum*serverVelocity[key]
    vel_agg_time = time.perf_counter() - vel_start

    return averaged_model, serverVelocity, vel_agg_time

def print_velocities(velocity, label="Velocity"):
    print(f"{label}:")
    for name, tensor in velocity.items():
        print(f" - {name}: shape={tensor.shape}, size={tensor.numel()}")
    print("=" * 50)

def federated(algo):
    start_time = time.time()
    num_clients = 5
    training_rounds = 15

    #***************DATASET CHOICE*******************
    trainSet = data_utils.load_mnist_dataset(isTrainDataset=True)
    #trainSet = data_utils.load_cifar10_dataset(isTrainDataset=True)
    testSet = data_utils.load_mnist_dataset(isTrainDataset=False)
    #testSet = data_utils.load_cifar10_dataset(isTrainDataset=False)

    #*************IID/NON-IID SPLIT*******************
    clientDatasets = data_utils.split_non_iid_class_proportional(trainSet, num_clients)
    testLoader = data_utils.get_dataloader(testSet)

    filename = os.path.join(output_dir, f'federated_metrics_cnn{algo}_test1.csv')
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "round", f"time_{algo}", f"accuracy_{algo}", f"loss_{algo}",
            "comm_uplink_bytes", "comm_downlink_bytes", "comm_total_bytes",
            "agg_time_s", "vel_agg_time_s",
        ])

    global client_models
    global client_velocities
    global client_distributions
    global client_grads

    serverModel = LogisticRegression().to(device)
    serverVelocity = {name: torch.zeros_like(param) for name, param in serverModel.named_parameters()}

    for rnd in range(training_rounds):
        print(f"Round {rnd+1} started")

        client_models.clear()
        client_velocities.clear()
        client_distributions.clear()
        client_data_sizes.clear()
        client_grads.clear()
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
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            t.start()
            client_threads.append(t)

        for t in client_threads:
            t.join()

        uplink_bytes, downlink_bytes, total_bytes = comm_bytes_per_round(
            algo, serverModel, serverVelocity, num_clients
        )

        agg_start = time.perf_counter()
        if algo == "fedavg":
            serverModel = fed_avg(client_models)
            vel_agg_time = 0.0
        elif algo == "mfl" or algo == "fednag":
            serverModel, serverVelocity, vel_agg_time = fed_momentum_nag(client_models, client_velocities)
        elif algo == "fedwan":
            serverModel, serverVelocity, vel_agg_time = fed_wan(client_models, client_velocities, client_distributions, rnd, num_clients)
        elif algo == "fedmom":
            serverModel, serverVelocity = fed_mom(client_models, serverModel, serverVelocity)
            vel_agg_time = 0.0
        elif algo == "mime":
            serverModel, serverVelocity, vel_agg_time = mime(client_models, client_grads, serverVelocity)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        agg_time = time.perf_counter() - agg_start

        testAcc, testLoss = test(serverModel, testLoader)
        curr_time = time.time() - start_time
        print(f"Round {rnd+1} done\tAccuracy: {testAcc:.4f}\tLoss: {testLoss:.4f}")
        print(f"  Comm — uplink: {uplink_bytes:,} B/client  downlink: {downlink_bytes:,} B/client  total: {total_bytes:,} B")
        print(f"  Agg time: {agg_time:.4f}s  (velocity agg: {vel_agg_time:.6f}s)")

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                rnd + 1, curr_time, testAcc, testLoss,
                uplink_bytes, downlink_bytes, total_bytes,
                round(agg_time, 6), round(vel_agg_time, 6),
            ])


if __name__ == "__main__":
    #federated('fedavg')
    federated('fedwan')
    federated('fednag')
    federated('mfl')
    federated('mime')
    federated('fedmom')
    merge()
