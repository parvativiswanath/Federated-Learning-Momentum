from collections import Counter
from copy import copy, deepcopy
from threading import Thread, Lock

import sys
print(sys.executable)
print(sys.version)

import torch
import time
import csv

from dataset import mnist_dataset
from model.layers import CNN, DNN, LogisticRegression
from model.train import test, train, train_with_momentum, train_with_NAG, train_mime, train_with_adan
from merger import merge

clientModels = []
clientVelocities = []
client_velocities = []
clientDistributions = []
client_data_sizes = []
clientGrads = []
emnist = False

clientModelsLock = Lock()
start_time = time.time()

clientNum = 5
trainingRounds = 20
momentum = 0.9

#***************DATASETS CHOICE*******************
trainSet = mnist_dataset.load_mnist_dataset(isTrainDataset=True)
#trainSet = mnist_dataset.load_cifar10_dataset(isTrainDataset=True)
#trainSet = mnist_dataset.load_emnist_dataset(isTrainDataset=True)
testSet = mnist_dataset.load_mnist_dataset(isTrainDataset=False)
#testSet = mnist_dataset.load_cifar10_dataset(isTrainDataset=False)

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
            #p = torch.tensor(count / sum(client_distribution.values()))
            p = torch.tensor(count)
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
    #client_data_size = len(clientDatasets[client][round])
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
    #client_data_size = len(clientDatasets[client][round])
    trainLoader = mnist_dataset.get_dataloader(clientTrainingSet)
    clientModel = deepcopy(serverModel)
    
    trainedClientModel, trainedClientGradients = train_mime(clientModel, trainLoader, serverVelocity)

    clientModelsLock.acquire()
    clientModels.append(trainedClientModel)
    client_data_sizes.append(client_data_size)
    clientGrads.append(trainedClientGradients)
    clientModelsLock.release()

    print(f"Client {client+1} done")

def clientTraining_with_Adan(serverModel, serverVelocity, clientDatasets, client):
    global clientModels
    global client_velocities
    global clientDistributions
    global client_data_sizes
    #clientTrainingSet = clientDatasets[client][round]
    clientTrainingSet = clientDatasets[client]
    client_data_size = len(clientDatasets[client])
    #client_data_size = len(clientDatasets[client][round])
    trainLoader = mnist_dataset.get_dataloader(clientTrainingSet)
    clientModel = deepcopy(serverModel)
    client_velocity = deepcopy(serverVelocity)

    trainedClientModel, trainedClientVelocity = train_with_adan(clientModel, trainLoader, client_velocity)

    clientdistribution = get_label_distribution(trainLoader)
    clientModelsLock.acquire()
    clientModels.append(trainedClientModel)
    client_velocities.append(trainedClientVelocity)
    clientDistributions.append(clientdistribution)
    client_data_sizes.append(client_data_size)
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
                global_distribution[label] = 800
            #global_distribution[label] += count

    
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
        global_velocity[key] = momentum * global_velocity[key] - 0.1 * weighted_diff[key]
        global_params[key] = global_params[key] + global_velocity[key]

    serverModel.load_state_dict(global_params)
    
    return serverModel, global_velocity


def mime(clientModels, clientGrads, serverVelocity):
    print('len:',len(clientModels))
    print('len:', len(clientGrads[0]))
    for i, client_grad in enumerate(clientGrads):
        print(f"Keys in clientGrads[{i}]: {client_grad.keys()}")
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
        if key not in clientGrads[0]:  # Check if the key exists in the gradient dictionary
            print(f"Skipping key {key} as it is not found in clientGrads.")
            continue
        #Average full batch gradients wrt to server parameters for each client dataset
        avg_clients_grads[key] = torch.stack([clientGrads[i][key].float() for i in range(len(clientGrads))], 0).mean(0)
        serverVelocity[key] = (1-momentum)*avg_clients_grads[key] + momentum*serverVelocity[key]

    return averagedModel, serverVelocity


def print_velocities(velocity, label="Velocity"):
    print(f"{label}:")
    for name, tensor in velocity.items():
        print(f" - {name}: shape={tensor.shape}, size={tensor.numel()}")
    print("=" * 50)

# class federatedConfig:
#     clientNum = 5
#     trainingRounds = 15

def federated(algo):
    # Open (or create) the CSV file and write headers
    #*********************POINT TO CHANGE****************
    #filename = 'federated_metrics_fedwan_non_iid_lr0.01.csv'
    filename = f'federated_metrics_base2.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"time_{algo}", f"accuracy_{algo}", f"loss_{algo}"])

    global clientModels
    global clientVelocities
    global clientDistributions
    global clientGrads
    config = federatedConfig()

    serverModel = CNN()
    serverVelocity = {name: torch.zeros_like(param) for name, param in serverModel.named_parameters()}

    for round in range(config.trainingRounds):
        print(f"Round {round+1} started")
        #clientDatasets = mnist_dataset.split_client_datasets(trainSet, config.clientNum, config.trainingRounds)
        #clientDatasets = mnist_dataset.test2(trainSet, config.clientNum)

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
        print('start_time: ',start_time,' curr time: ',time.time())

         # Append the results to the CSV file
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round + 1, curr_time, testAcc, testLoss])

        print(f"Round {round+1} done\tAccuracy: {testAcc}\tLoss: {testLoss}")


# if __name__ == "__main__":
#     start_time = time.time()
#     #federated('fedavg')
#     federated('fedwan')
#     #federated('fednag')
#     #federated('mfl')
#     #federated('mime')
#     #federated('fedmom')
#     #merge()



def fedAdan(clientModels, client_velocities):
    print('Number of Clients:', len(client_velocities),len(client_velocities[0]))
    for name, param in client_velocities[0].items():
        print(name)
    
    averagedModel = deepcopy(clientModels[0])

    averagedMomentum = {}
    averagedVelocity = {}
    averagedN = {}

    for name, param in client_velocities[0].items():
        if isinstance(param, dict) and 'exp_avg' in param:
            averagedMomentum[name] = torch.zeros_like(param['exp_avg'])
            averagedVelocity[name] = torch.zeros_like(param['exp_avg_diff'])
            averagedN[name] = torch.zeros_like(param['exp_avg_sq'])
        else:
            print(f"Skipping {name}, unexpected structure in initialization: {param}")

    with torch.no_grad():
        # Average the model parameters
        for model in clientModels[1:]:
            for param1, param2 in zip(averagedModel.parameters(), model.parameters()):
                param1.data += param2.data
        for param in averagedModel.parameters():
            param.data /= len(clientModels)

        # Aggregate Adan-style velocities
        for velocity in client_velocities:
            for name in averagedVelocity:
                averagedMomentum[name] += velocity[name]['exp_avg']
                averagedVelocity[name] += velocity[name]['exp_avg_diff']
                averagedN[name] += velocity[name]['exp_avg_sq']
                

        # Final averaging across clients
        num_clients = len(client_velocities)
        for name in averagedMomentum:
            averagedMomentum[name] /= num_clients
            averagedVelocity[name] /= num_clients
            averagedN[name] /= num_clients

    return averagedModel, {
        name: {
            'exp_avg': averagedMomentum[name],
            'exp_avg_diff': averagedVelocity[name],
            'exp_avg_sq': averagedN[name]
        } for name in averagedMomentum
    }



class federatedConfig:
    clientNum = 5
    trainingRounds = 40


def federated_adan():
    filename = f'federated_metrics_adantest2.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time_fedadan", "accuracy_fedadan", "loss_fedadan"])

    global clientModels, client_velocities, clientDistributions, client_data_sizes, clientGrads
    config = federatedConfig()

    serverModel = CNN()
    server_velocity = {}
    for name, param in serverModel.named_parameters():
        server_velocity[name] = {
            'exp_avg': torch.zeros_like(param),
            'exp_avg_diff': torch.zeros_like(param),
            'exp_avg_sq': torch.zeros_like(param)
        }

    for round in range(config.trainingRounds):
        print(f"Round {round+1} started")

        clientModels.clear()
        client_velocities.clear()
        clientDistributions.clear()
        client_data_sizes.clear()
        clientGrads.clear()
        clientThreads = []
        for client in range(config.clientNum):
            t = Thread(
                    target=clientTraining_with_Adan, 
                    args=(serverModel, server_velocity, clientDatasets, client)
                )
            t.start()
            clientThreads.append(t)

        for t in clientThreads:
            t.join()

        print('Printing server model')
        print(serverModel)

        # FedAdan aggregation logic
        serverModel, server_velocity = fedAdan(clientModels, client_velocities)

        testAcc, testLoss = test(serverModel, testLoader)
        print(f"Round {round+1} done\tAccuracy: {testAcc}\tLoss: {testLoss}")
        curr_time = time.time() - start_time
        print(f'start_time: {start_time}, curr time: {time.time()}')

        # Append results to CSV
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round + 1, curr_time, testAcc, testLoss])

        print(f"Round {round+1} done\tAccuracy: {testAcc}\tLoss: {testLoss}")



if __name__ == "__main__":
    start_time = time.time()
    federated('fednag')
    start_time = time.time()
    federated_adan()
    
