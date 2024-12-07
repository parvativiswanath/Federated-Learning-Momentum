from collections import Counter
from copy import copy, deepcopy
from threading import Thread, Lock

import torch
import time
import csv

from dataset import mnist_dataset
from model.layers import CNN, DNN, LogisticRegression
from model.train import test, train, train_with_momentum, train_with_NAG, train_mime
from merger import merge

clientModels = []
clientVelocities = []
clientDistributions = []
client_data_sizes = []
clientGrads = []
emnist = False

clientModelsLock = Lock()
start_time = time.time()

clientNum = 5
trainingRounds = 30
momentum = 0.9

#***************DATASETS CHOICE*******************
trainSet = mnist_dataset.load_mnist_dataset(isTrainDataset=True)
#trainSet = mnist_dataset.load_emnist_dataset(isTrainDataset=True)
testSet = mnist_dataset.load_mnist_dataset(isTrainDataset=False)

#*************IID/NON-IID SPLIT*******************
#clientDatasets = mnist_dataset.split_client_datasets(trainSet, clientNum, trainingRounds)
# clientDatasets = mnist_dataset.split_client_datasets_non_iid(trainSet, clientNum, trainingRounds)
clientDatasets = mnist_dataset.test2(trainSet, clientNum)
testLoader = mnist_dataset.get_dataloader(testSet)

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
            p = torch.tensor(count / sum(client_distribution.values()))
            q = global_distribution[label]
            if p is not None and q is not None:
                kl_divergence += p * torch.log(p / q)
        return kl_divergence



def clientTraining(serverModel, clientDatasets, client, round):
    global clientModels
    #clientTrainingSet = clientDatasets[client][round]
    clientTrainingSet = clientDatasets[client]
    trainLoader = mnist_dataset.get_dataloader(clientTrainingSet)
    clientModel = deepcopy(serverModel)
    trainedClientModel = train(clientModel, trainLoader)
    clientModelsLock.acquire()
    clientModels.append(trainedClientModel)
    clientModelsLock.release()

    print(f"Client {client+1} done")

def clientTraining_withLocalMomentum(serverModel, serverVelocity, clientDatasets, client, round, nesterov):
    global clientModels
    global clientVelocities
    global clientDistributions
    global client_data_sizes
    #clientTrainingSet = clientDatasets[client][round]
    clientTrainingSet = clientDatasets[client]
    client_data_size = len(clientDatasets[client])
    trainLoader = mnist_dataset.get_dataloader(clientTrainingSet)
    clientModel = deepcopy(serverModel)
    clientVelocity = deepcopy(serverVelocity)
    if nesterov:
        trainedClientModel, trainedClientVelocity = train_with_NAG(clientModel, trainLoader, clientVelocity)
    else:
        trainedClientModel, trainedClientVelocity = train_with_momentum(clientModel, trainLoader, clientVelocity)
    clientdistribution = get_label_distribution(trainLoader)
    clientModelsLock.acquire()
    clientModels.append(trainedClientModel)
    clientVelocities.append(trainedClientVelocity)
    clientDistributions.append(clientdistribution)
    client_data_sizes.append(client_data_size)
    clientModelsLock.release()

    print(f"Client {client+1} done")

def clientTraining_Mime(serverModel, serverVelocity, clientDatasets, client, round):
    global clientModels
    global client_data_sizes
    global clientGrads
    #clientTrainingSet = clientDatasets[client][round]
    clientTrainingSet = clientDatasets[client]
    client_data_size = len(clientDatasets[client])
    trainLoader = mnist_dataset.get_dataloader(clientTrainingSet)
    clientModel = deepcopy(serverModel)
    
    trainedClientModel, trainedClientGradients = train_mime(clientModel, trainLoader, serverVelocity)

    clientModelsLock.acquire()
    clientModels.append(trainedClientModel)
    client_data_sizes.append(client_data_size)
    clientGrads.append(trainedClientGradients)
    clientModelsLock.release()

    print(f"Client {client+1} done")


def fedAvg(clientModels):
    print('NO OF CLIENT MODELS RECEIVED: ', len(clientModels))
    averagedModel = deepcopy(clientModels[0])
    with torch.no_grad():
        for model in clientModels[1:]:
            for param1, param2 in zip(averagedModel.parameters(), model.parameters()):
                param1.data += param2.data
        for param in averagedModel.parameters():
            param.data /= len(clientModels)
    return averagedModel


def fedmomentum_NAG(clientModels,clientVelocities):
    averagedModel = deepcopy(clientModels[0])
    averagedVelocity = deepcopy(clientVelocities[0])
    with torch.no_grad():
        # Average the models
        for model in clientModels[1:]:
            for param1, param2 in zip(averagedModel.parameters(), model.parameters()):
                param1.data += param2.data
        for param in averagedModel.parameters():
            param.data /= len(clientModels)
        # Average the velocities
        for velocity in clientVelocities[1:]:
            for name in averagedVelocity:
                averagedVelocity[name] += velocity[name]  
        for name in averagedVelocity:
            averagedVelocity[name] /= len(clientVelocities) 
    # print(averagedModel)  
    return averagedModel, averagedVelocity

def fedWAN(clientModels,clientVelocities, clientDistributions, round, num_clients):
    averagedVelocity = deepcopy(clientVelocities[0])
    averagedModel = deepcopy(clientModels[0])

    global client_data_sizes
    print('client_data_sizes: ',client_data_sizes)
    print('CLIENT DISTRIBUTIONS:')
    print(clientDistributions)
    total_size = sum(client_data_sizes)
    global_distribution = {}
    for distribution in clientDistributions:
        for label, count in distribution.items():
            if label not in global_distribution:
                global_distribution[label] = 0
            global_distribution[label] += count

    
    print('GLOBAL DISTRIBUTION:')
    print(global_distribution)
    # Normalize the distribution
    for label, count in global_distribution.items():
        global_distribution[label] /= total_size
    
    #Initialise weights
    weights = torch.zeros(len(clientDistributions), dtype=torch.float)

    # Calculate weights
    for client in range(len(clientDistributions)):
        print("client: ",client)
        print("client_distributions: ",clientDistributions[client])
        kli = calculate_kl_divergence(clientDistributions[client], global_distribution)
        print("kli: ",kli)
        weights[client] = 1/kli
    print('weights before normalizing: ', weights)
    total_weight = weights.sum()
    for client in range(len(clientDistributions)):
        weights[client] = weights[client] / total_weight
    print('weights for aggregation: ', weights)

    with torch.no_grad():
        # Initialize model parameters to zero
        for param in averagedModel.parameters():
            param.data.zero_()
        
        # Weighted aggregation of models
        for model, weight in zip(clientModels, weights):
            for param_avg, param_client in zip(averagedModel.parameters(), model.parameters()):
                param_avg.data += weight * param_client.data

    # Average the velocities
    for velocity in clientVelocities[1:]:
        for name in averagedVelocity:
            averagedVelocity[name] += velocity[name]  
    for name in averagedVelocity:
        averagedVelocity[name] /= len(clientVelocities) 

    return averagedModel, averagedVelocity
    

def fedmom(clientModels, serverModel, globalVelocity):
    global_params = serverModel.state_dict()
    weighted_diff = {key: torch.zeros_like(tensor) for key, tensor in global_params.items()}
    global_velocity = {key: torch.zeros_like(tensor) for key, tensor in global_params.items()}

    for clientModel in clientModels:
        client_state = clientModel.state_dict()
        for key in global_params.keys():
            weighted_diff[key] += (global_params[key] - client_state[key])
    
    for key in global_params.keys():
        # global_velocity[key] = global_params[key] - weighted_diff[key]
        # global_params[key] = global_velocity[key] + momentum * (global_velocity[key] - globalVelocity[key])
        global_velocity[key] = momentum * global_velocity[key] - 0.6 * weighted_diff[key]
        global_params[key] = global_params[key] + global_velocity[key]

    serverModel.load_state_dict(global_params)
    
    return serverModel, global_velocity


def mime(clientModels, clientGrads, serverVelocity):
    print('len:',len(clientModels))
    averagedModel = deepcopy(clientModels[0])
    with torch.no_grad():
        for model in clientModels[1:]:
            for param1, param2 in zip(averagedModel.parameters(), model.parameters()):
                param1.data += param2.data
        for param in averagedModel.parameters():
            param.data /= len(clientModels)

    #server velocity
    #Average model parameters
    global_dict = averagedModel.state_dict()
    avg_clients_grads = deepcopy(global_dict)
    for key in global_dict.keys():
        #Average full batch gradients wrt to server parameters for each client dataset
        avg_clients_grads[key] = torch.stack([clientGrads[i][key].float() for i in range(len(clientGrads))], 0).mean(0)
        serverVelocity[key] = (1-momentum)*avg_clients_grads[key] + momentum*serverVelocity[key]

    return averagedModel, serverVelocity

class federatedConfig:
    clientNum = 5
    trainingRounds = 30

def print_velocities(velocity, label="Velocity"):
    print(f"{label}:")
    for name, tensor in velocity.items():
        print(f" - {name}: shape={tensor.shape}, size={tensor.numel()}")
    print("=" * 50)

def federated(algo):
    # Open (or create) the CSV file and write headers
    #*********************POINT TO CHANGE****************
    #filename = 'federated_metrics_fedwan_non_iid_lr0.01.csv'
    filename = f'federated_metrics_{algo}_non_iid_lr0.001.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"time_{algo}", f"accuracy_{algo}", f"loss_{algo}"])

    global clientModels
    global clientVelocities
    global clientDistributions
    global clientGrads
    config = federatedConfig()

    serverModel = DNN()
    serverVelocity = {name: torch.zeros_like(param) for name, param in serverModel.named_parameters()}

    for round in range(config.trainingRounds):
        print(f"Round {round+1} started")

        clientModels.clear()
        clientVelocities.clear()
        clientDistributions.clear()
        client_data_sizes.clear()
        clientGrads.clear()
        clientThreads = []
        for client in range(config.clientNum):
            a = 6
            if algo == "fedavg":
                t = Thread(
                    target=clientTraining, 
                    args=(serverModel, clientDatasets, client, round)
                )
            elif algo == "mfl":
                t = Thread(
                    target=clientTraining_withLocalMomentum, 
                    args=(serverModel, serverVelocity, clientDatasets, client, round, False)
                )
            elif algo == "fednag":
                t = Thread(
                    target=clientTraining_withLocalMomentum, 
                    args=(serverModel, serverVelocity, clientDatasets, client, round, True)
                )
            elif algo == "fedwan":
                t = Thread(
                    target=clientTraining_withLocalMomentum, 
                    args=(serverModel, serverVelocity, clientDatasets, client, round, True)
                )
            elif algo == "fedmom":
                t = Thread(
                    target=clientTraining,
                    args=(serverModel, clientDatasets, client, round)
                )
            elif algo == "mime":
                t = Thread(
                    target=clientTraining_Mime,
                    args=(serverModel, serverVelocity, clientDatasets, client, round)
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo}")
            
            t.start()
            clientThreads.append(t)


        for t in clientThreads:
            t.join()

        print('printing server model')
        print(serverModel)  
        
        if algo == "fedavg":
            serverModel = fedAvg(clientModels)
        elif algo == "mfl" or algo == "fednag":
            serverModel, serverVelocity = fedmomentum_NAG(clientModels, clientVelocities)
        elif algo == "fedwan":
            serverModel, serverVelocity = fedWAN(clientModels, clientVelocities, clientDistributions, round, config.clientNum)
        elif algo == "fedmom":
            serverModel, serverVelocity = fedmom(clientModels, serverModel, serverVelocity)
        elif algo == "mime":
            serverModel, serverVelocity = mime(clientModels, clientGrads, serverVelocity)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")


        testAcc, testLoss = test(serverModel, testLoader)
        print(f"Round {round+1} done\tAccuracy: {testAcc}\tLoss: {testLoss}")
        curr_time = time.time() - start_time

         # Append the results to the CSV file
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round + 1, curr_time, testAcc, testLoss])

        print(f"Round {round+1} done\tAccuracy: {testAcc}\tLoss: {testLoss}")


if __name__ == "__main__":
    start_time = time.time()
    federated('fedavg')
    federated('mfl')
    federated('fedwan')
    federated('fednag')
    federated('mime')
    federated('fedmom')
    merge()
