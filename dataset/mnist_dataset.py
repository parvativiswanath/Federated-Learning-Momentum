from collections import defaultdict
import os
import random

import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


def load_mnist_dataset(isTrainDataset=True) -> Dataset:
    mnistDataset = datasets.MNIST(
        os.path.dirname(os.path.realpath(__file__)) + "/data",
        train=isTrainDataset,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    return mnistDataset

def load_emnist_dataset(isTrainDataset=True) -> Dataset:
    emnistDataset = datasets.EMNIST(
        os.path.dirname(os.path.realpath(__file__)) + "/data",
        split="byclass",
        train=isTrainDataset,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    return emnistDataset


def split_client_datasets(dataset, clientNum, roundNum):
    countPerSet = len(dataset) // (clientNum * roundNum)
    clientDatasets = [[] for _ in range(clientNum)]
    print('Length of client datasets')
    for client in range(clientNum):
        for round in range(roundNum):
            low = countPerSet * (round + client * roundNum)
            high = low + countPerSet
            subsetIndices = [i for i in range(low, high)]
            clientDatasets[client].append(Subset(dataset, subsetIndices))
            print(f'len(clientDatasets[{client}][{round}]: ',len(clientDatasets[client][round]))

    return clientDatasets

def split_client_datasets_non_iid(dataset, clientNum, roundNum):
    total_samples = len(dataset)
    if total_samples < clientNum * roundNum:
        raise ValueError("Not enough samples in the dataset to distribute among all clients and rounds.")
    
    # Shuffle dataset indices once to ensure randomness
    dataset_indices = list(range(total_samples))
    random.shuffle(dataset_indices)
    
    # Calculate number of samples per round
    samples_per_round = total_samples // roundNum
    remaining_samples = total_samples % roundNum
    
    # Split dataset into rounds
    round_indices = []
    start = 0
    for i in range(roundNum):
        end = start + samples_per_round + (1 if i < remaining_samples else 0)
        round_indices.append(dataset_indices[start:end])
        start = end
    
    # Allocate data to clients within each round
    clientDatasets = [[] for _ in range(clientNum)]
    for round, indices in enumerate(round_indices):
        random.shuffle(indices)  # Shuffle indices for the round
        remaining_indices = indices.copy()
        
        for client in range(clientNum):
            if client == clientNum - 1:
                # Last client gets all remaining indices
                client_data_indices = remaining_indices
            else:
                # Randomly decide the number of samples for the client in this round
                max_samples = len(remaining_indices) // (clientNum - client)
                num_samples = random.randint(1, max_samples)
                client_data_indices = remaining_indices[:num_samples]
                remaining_indices = remaining_indices[num_samples:]  # Update remaining indices
            
            clientDatasets[client].append(Subset(dataset, client_data_indices))
            print(f'len(clientDatasets[{client}][{round}]): ', len(clientDatasets[client][round]))
    
    return clientDatasets


def generate_random_class_distribution_mnist(clientNum, num_classes=10, total_samples_per_class={0: 5923,1: 6742,2: 5958,3: 6131,4: 5842,5: 5421,6: 5918,7: 6265,8: 5851,9: 5949}, min_samples_per_class=10, max_samples_per_class=2000):
    class_distributions = []
    
    for _ in range(clientNum):
        client_dist = {}
        for class_label in range(num_classes):
            max_samples = min(max_samples_per_class, total_samples_per_class[class_label])
            if max_samples < min_samples_per_class:
                raise ValueError(f"Not enough samples available for class {class_label} to meet constraints.")
            client_dist[class_label] = random.randint(min_samples_per_class, max_samples)
            total_samples_per_class[class_label] -= client_dist[class_label]
        class_distributions.append(client_dist)
    
    return class_distributions

def generate_random_class_distribution_mnist(clientNum, num_classes=10, total_samples_per_class={0: 5923,1: 6742,2: 5958,3: 6131,4: 5842,5: 5421,6: 5918,7: 6265,8: 5851,9: 5949}, min_samples_per_class=10, max_samples_per_class=2000):
    class_distributions = []
    
    for _ in range(clientNum):
        client_dist = {}
        for class_label in range(num_classes):
            max_samples = min(max_samples_per_class, total_samples_per_class[class_label])
            if max_samples < min_samples_per_class:
                raise ValueError(f"Not enough samples available for class {class_label} to meet constraints.")
            client_dist[class_label] = random.randint(min_samples_per_class, max_samples)
            total_samples_per_class[class_label] -= client_dist[class_label]
        class_distributions.append(client_dist)
    
    return class_distributions

def generate_random_class_distribution_emnist(
    clientNum, 
    num_classes=62, 
    total_samples_per_class={
        i: v for i, v in enumerate([
            4800, 5843, 5542, 5293, 5583, 5076, 4861, 5100, 5156, 4831, 4682, 5095, 4930, 4838, 
            4932, 4900, 4965, 4929, 4773, 4798, 4646, 4730, 4687, 4699, 4761, 4659, 4696, 4669, 
            4753, 4687, 4605, 4650, 4556, 4572, 4558, 4545, 4582, 4592, 4576, 4587, 4562, 4521, 
            4538, 4529, 4516, 4519, 4510, 4508, 4512, 4514, 4515, 4501, 4502, 4516, 4520, 4511, 
            4507, 4505, 4504, 4502, 4500, 4500
        ])
    },
    min_samples_per_class=10, 
    max_samples_per_class=2000
):
    class_distributions = []
    
    for _ in range(clientNum):
        client_dist = {}
        for class_label in range(num_classes):
            max_samples = min(max_samples_per_class, total_samples_per_class[class_label])
            if max_samples < min_samples_per_class:
                raise ValueError(f"Not enough samples available for class {class_label} to meet constraints.")
            client_dist[class_label] = random.randint(min_samples_per_class, max_samples)
            total_samples_per_class[class_label] -= client_dist[class_label]
        class_distributions.append(client_dist)
    
    return class_distributions

def split_non_iid_client_datasets(dataset, clientNum, roundNum,emnist):
    if emnist:
        class_distributions = generate_random_class_distribution_emnist(clientNum)
    else:
        class_distributions = generate_random_class_distribution_mnist(clientNum)
    print('class distributions for all clients are - ')
    print(class_distributions)
    # Step 1: Organize dataset by class
    data_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):  # Assuming dataset is iterable and returns (data, label)
        data_by_class[label].append(idx)

    # Shuffle data within each class
    for label in data_by_class:
        random.shuffle(data_by_class[label])

    # Step 2: Allocate data to clients based on specified distributions
    client_data = [[] for _ in range(clientNum)]
    
    for client_id, class_dist in enumerate(class_distributions):
        for class_label, count in class_dist.items():
            allocated_samples = data_by_class[class_label][:count]
            client_data[client_id].extend(allocated_samples)
            data_by_class[class_label] = data_by_class[class_label][count:]  # Remove allocated samples

    # Step 3: Divide each client's data into rounds
    client_datasets = [[] for _ in range(clientNum)]
    for client_id in range(clientNum):
        client_samples = client_data[client_id]
        random.shuffle(client_samples)  # Shuffle to mix classes
        count_per_round = len(client_samples) // roundNum

        for round_id in range(roundNum):
            low = count_per_round * round_id
            high = low + count_per_round
            if round_id == roundNum - 1:  # Include any leftover samples in the last round
                high = len(client_samples)
            round_indices = client_samples[low:high]
            client_datasets[client_id].append(Subset(dataset, round_indices))

    return client_datasets


def get_dataloader(dataset, batchSize=64):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batchSize, shuffle=True, drop_last=True
    )
    return dataloader
