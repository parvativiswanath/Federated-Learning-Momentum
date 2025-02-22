from copy import deepcopy
from threading import Event, Lock
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from flask import Flask, request, send_file
import torch
import socket
import re
import time


app = Flask(__name__)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 3)

    def forward(self, x):
        #x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
global_model = Net()
print('Global Model')
print(global_model)
#test_loader = DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])), batch_size=1000, shuffle=True)
criterion = nn.CrossEntropyLoss()
global_velocity = {name:torch.zeros_like(param) for name,param in global_model.named_parameters()}
#averaging_in_progress = False


#MULTIPLE CLIENTS TEST
lock = Lock()
event = Event()

# Temporary storage for model parameters from clients
client_models = []
client_velocities = []
client_distributions = []

num_clients = 2


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     global client_models, global_model, client_waiting_responses, averaging_in_progress, rnd

#     # Receive the file
#     uploaded_file = request.files['file']
#     client_model_path = 'client_model.pth'
#     uploaded_file.save(client_model_path)

#     # Load the parameters from the client model
#     client_model = torch.load(client_model_path)

#     with lock:
#         # Store the client model and append this client to the waiting list
#         client_models.append(client_model)
#         print('client model appended')
#         #client_waiting_responses.append(request)

#         # Perform averaging if enough clients have uploaded models
#         if len(client_models) >= 1:
#             print('averaging')
#             #averaging_in_progress = True  # Lock the averaging process for this thread

#             # Perform FedAvg
#             average_model_parameters(global_model, client_models)
#             print(f'\nROUND {rnd}')
#             rnd+=1
#             test(global_model, test_loader)
#             #averaging_in_progress = False

#             # Clear client_models for the next round
#             client_models = []

#             # Save the updated model parameters
#             updated_model_path = 'updated_model_params.pth'
#             torch.save(global_model.state_dict(), updated_model_path)

#             event.set()
#             print('event set')

#             time.sleep(3)
#             event.clear()

#             print('client which averaged: sending response')
#             return send_file('updated_model_params.pth', as_attachment=True)

#     # # If the condition is still not met, wait for more uploads
#     # if len(client_models) < 3:
#     #     event.wait()

#     print('sending response')
#     return send_file('updated_model_params.pth', as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    global client_models, client_velocities, client_distributions,global_model, rnd

    # Receive the files
    file1 = request.files['file1']
    file2 = request.files['file2']
    file3 = request.files['file3']

    print('client files received')
    # Save the files locally
    client_model_path = 'client_model.pth'
    client_velocity_path = 'client_velocity.pth'
    client_distribution_path = 'client_distribution.pth'
    file1.save(client_model_path)
    file2.save(client_velocity_path)
    file3.save(client_distribution_path)

    # Load the parameters from the client model and velocity
    client_model = torch.load(client_model_path)
    client_velocity = torch.load(client_velocity_path)
    client_distribution = torch.load(client_distribution_path)
    print('client distribution is: ',client_distribution)

    with lock:
        # Store the client model and velocity
        client_models.append(client_model)
        client_velocities.append(client_velocity)
        client_distributions.append(client_distribution)
        print('Client model and velocity appended')

        # Perform averaging if enough clients have uploaded models
        if len(client_models) >= num_clients:  
            print('Averaging models and velocities')
            
            # Perform FedAvg for models
            average_model_parameters_with_weights(global_model, client_models, client_distributions)
            #average_model_parameters(global_model, client_models)

            # Average velocities 
            for velocity in client_velocities[1:]:
                for name in global_velocity:
                    global_velocity[name] += velocity[name]  
            for name in global_velocity:
                global_velocity[name] /= len(client_velocities) 

            print(f'\nROUND {rnd}')
            rnd += 1
            #test(global_model, test_loader)  
            
            # Clear client data for the next round
            client_models = []
            client_velocities = []
            client_distributions = []

            # Save the updated model parameters and velocity
            updated_model_path = 'updated_model_params.pth'
            updated_velocity_path = 'updated_velocity.pth'
            torch.save(global_model.state_dict(), updated_model_path)
            torch.save(global_velocity, updated_velocity_path)

            # Create a ZIP file to send back to the client
            zip_path = 'updated_files.zip'
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(updated_model_path)
                zipf.write(updated_velocity_path)
            
            event.set()
            print('event set')

            time.sleep(3)
            event.clear()

            print('client which averaged: sending response as zip file')
            return send_file(zip_path, as_attachment=True)

    # If the condition is still not met, wait for more uploads
    if len(client_models) < num_clients:
        event.wait()

    print('sending response as zip file')
    return send_file('updated_model_params.pth', as_attachment=True)



def average_model_parameters(global_model, client_models):

    # Initialize a list of model parameters (keys and values)
    model_state_dict = global_model.state_dict()

    # Calculate the average
    for param_name in model_state_dict:
        client_params = [client_model[param_name] for client_model in client_models]
        averaged_param = torch.stack(client_params).mean(dim=0)

        model_state_dict[param_name] = averaged_param

    # Update the global model with the new averaged parameters
    global_model.load_state_dict(model_state_dict)
    

def calculate_kl_divergence(client_distribution, global_distribution):
        kl_divergence = 0.0
        for label, count in client_distribution.items():
            #p = torch.tensor(count / sum(client_distribution.values()))
            p = torch.tensor(count)
            q = global_distribution[label]
            if p is not None and q is not None:
                kl_divergence += p * torch.log(p / q)
        return kl_divergence

def average_model_parameters_with_weights(global_model,client_models,client_distributions):
    # averagedModel = deepcopy(global_model)
    model_state_dict = global_model.state_dict()
    #model_state_dict = global_model.parameters()

    # global client_data_sizes
    # print('client_data_sizes: ',client_data_sizes)
    print('CLIENT DISTRIBUTIONS:')
    print(client_distributions)
    total_size = 800 * num_clients
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
    weights = torch.zeros(len(client_distributions), dtype=torch.float)

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

    # with torch.no_grad():
    #     # Initialize model parameters to zero
    #     for param in model_state_dict:
    #         param.data.zero_()
        
    #     # Weighted aggregation of models
    #     for model, weight in zip(client_models, weights):
    #         for param_avg, param_client in zip(model_state_dict, model.parameters()):
    #             param_avg.data += weight * param_client.data

    # global_model.load_state_dict(model_state_dict)

    with torch.no_grad():
        # Initialize model parameters to zero
        # for param_name in model_state_dict:
        #     model_state_dict[param_name] = 0
        
        for param_name in model_state_dict:
            averaged_param = 0
            for client_model,weight in zip(client_models,weights):
                averaged_param += weight * client_model[param_name]

            model_state_dict[param_name] = averaged_param

    global_model.load_state_dict(model_state_dict)
    
# Test the global model
def test(global_model, test_loader):
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = global_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    # accuracy_append(correct / len(test_loader.dataset))
    # loss_history.append(test_loss)
    # time_history.append(time.time() - start_time)



rnd=1
app.run(host='0.0.0.0', port=5000, threaded=True)

    