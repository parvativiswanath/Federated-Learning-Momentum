from threading import Event, Lock
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
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
global_model = Net()
test_loader = DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])), batch_size=1000, shuffle=True)
criterion = nn.CrossEntropyLoss()
averaging_in_progress = False


#MULTIPLE CLIENTS TEST
lock = Lock()
event = Event()

# Temporary storage for model parameters from clients
client_models = []
client_waiting_responses = []


@app.route('/upload', methods=['POST'])
def upload_file():
    global client_models, global_model, client_waiting_responses, averaging_in_progress, rnd

    # Receive the file
    uploaded_file = request.files['file']
    client_model_path = 'client_model.pth'
    uploaded_file.save(client_model_path)

    # Load the parameters from the client model
    client_model = torch.load(client_model_path)

    with lock:
        # Store the client model and append this client to the waiting list
        client_models.append(client_model)
        print('client model appended')
        #client_waiting_responses.append(request)

        # Perform averaging if enough clients have uploaded models
        if len(client_models) >= 3:
            print('averaging')
            #averaging_in_progress = True  # Lock the averaging process for this thread

            # Perform FedAvg
            average_model_parameters(global_model, client_models)
            print(f'\nROUND {rnd}')
            rnd+=1
            test(global_model, test_loader)
            #averaging_in_progress = False

            # Clear client_models for the next round
            client_models = []

            # Save the updated model parameters
            updated_model_path = 'updated_model_params.pth'
            torch.save(global_model.state_dict(), updated_model_path)

            event.set()
            print('event set')

            time.sleep(3)
            event.clear()

            print('client which averaged: sending response')
            return send_file('updated_model_params.pth', as_attachment=True)

    # If the condition is still not met, wait for more uploads
    if len(client_models) < 3:
        event.wait()

    print('sending response')
    return send_file('updated_model_params.pth', as_attachment=True)



def average_model_parameters(global_model, client_models):
    """
    Perform Federated Averaging by averaging the parameters of the models.
    """
    # Initialize a list of model parameters (keys and values)
    model_state_dict = global_model.state_dict()

    # For each parameter in the model, calculate the average across all clients
    for param_name in model_state_dict:
        # Get the parameter from each client model
        client_params = [client_model[param_name] for client_model in client_models]

        # Average the client parameters (assuming equal weight for each client)
        averaged_param = torch.stack(client_params).mean(dim=0)

        # Update the global model with the averaged parameter
        model_state_dict[param_name] = averaged_param

    # Update the global model with the new averaged parameters
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

