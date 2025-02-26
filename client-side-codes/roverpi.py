import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from collections import Counter
import pandas as pd
import numpy as np
#import socket
import requests
import zipfile
import os
from torch.utils.data import DataLoader, TensorDataset

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# # Downloading MNIST dataset (train and test)
# trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model on paddy field data")
parser.add_argument("filename", type=str, help="CSV file containing training data")
args = parser.parse_args()

# Load the dataset
data = pd.read_csv(args.filename)

# Ensure data has the expected columns
assert set(['Water Level', 'Moisture Percentage', 'Light Intensity', 'Temperature', 'Humidity', 'Cluster']).issubset(data.columns), "Missing required columns"

# Load the dataset
test_data = pd.read_csv('paddyfield_test.csv')

# Ensure data has the expected columns
assert set(['Water Level', 'Moisture Percentage', 'Light Intensity', 'Temperature', 'Humidity', 'Cluster']).issubset(test_data.columns), "Missing required columns"

# Separate features and labels for training
features_train = data[['Water Level', 'Moisture Percentage', 'Light Intensity', 'Temperature', 'Humidity']].values
labels_train = data['Cluster'].values

# Separate features and labels for testing
features_test = test_data[['Water Level', 'Moisture Percentage', 'Light Intensity', 'Temperature', 'Humidity']].values
labels_test = test_data['Cluster'].values

# Normalize features
mean = np.mean(features_train, axis=0)
std = np.std(features_train, axis=0)
features_train = (features_train - mean) / std
features_test = (features_test - mean) / std  # Use same mean and std from training set

# Convert to tensors
features_train_tensor = torch.tensor(features_train, dtype=torch.float32)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)
features_test_tensor = torch.tensor(features_test, dtype=torch.float32)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(features_train_tensor, labels_train_tensor)
test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Defining a Simple Neural Network (Feedforward)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # Input layer: 28x28 pixels -> 128 neurons
        self.fc2 = nn.Linear(10, 5)       # Hidden layer: 128 -> 64 neurons
        self.fc3 = nn.Linear(5, 3)        # Output layer: 64 -> 10 neurons (for 10 classes)

    def forward(self, x):
        #x = x.view(-1, 28 * 28)  # Flatten 28x28 images to a 1D vector
        x = torch.relu(self.fc1(x))  # ReLU activation for first hidden layer
        x = torch.relu(self.fc2(x))  # ReLU activation for second hidden layer
        x = self.fc3(x)             # Output layer (no activation, raw logits)
        return x

def get_label_distribution(data_loader):
        label_counts = Counter()

        # Count label occurrences
        for _, labels in data_loader:
            label_counts.update(labels.tolist())

        # Normalize to get the distribution
        distribution = {label: count for label, count in label_counts.items()}
        return distribution

def send_params():
    url = 'http://192.168.67.103:5000/upload'
        
    # Paths to the files you want to upload
    file1_path = 'local_params.pth'
    file2_path = 'local_velocity.pth'
    file3_path = "local_distribution.pth"

    # Open the files in binary mode
    with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2, open(file3_path, 'rb') as file3:
        # Prepare the files for upload
        files = {
            'file1': file1,
            'file2': file2,
            'file3': file3
        }

        # Send the POST request
        print('\nUploading files...')
        response = requests.post(url, files=files) 

        # Handle the server's response
        if response.status_code == 200:
            # Save the ZIP file sent by the server
            zip_path = 'received_models.zip'
            with open(zip_path, 'wb') as zip_file:
                zip_file.write(response.content)

            # Extract the contents of the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('received_models')

            # Clean up the ZIP file if necessary
            os.remove(zip_path)

            print('Files received and extracted:')
            print(os.listdir('received_models'))  # List extracted files
            file_list = os.listdir('received_models')
            
            # os.rename(file_list[0],"updated_model_parameters.pth")
            # os.rename(file_list[1],"updated_velocity.pth")
        else:
            print(f'Error: Server responded with status code {response.status_code}') 
       
    # Load the updated parameters into your model
    model.load_state_dict(torch.load('received_models/updated_model_params.pth'))  
    velocity = torch.load('received_models/updated_velocity.pth')
    return model,velocity

def train_model(model,velocity):

    # Load the state_dict from file
    # global_state_dict = torch.load("global_params.pth")

    # Load the state_dict into the model
    # model.load_state_dict(global_state_dict)
   
   
    # 3. Model, Loss Function, and Optimizer
    criterion = nn.CrossEntropyLoss()  # Cross entropy for multi-class classification
    #optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent
    learningRate = 0.001
    momentum = 0.9
    epochs = 2  # Number of epochs to train the model

    model.train()
    print("\nTraining:")
    for epoch in range(epochs):

        epochLoss = 0
        for input, target in trainloader:

            # Reset such that only gradients that pertain
            # to the current input are used
            model.zero_grad()

            # Forward
            output = model(input)

            # No need to shape target to one-hot encoding
            loss = criterion(output, target)
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Update velocity using momentum
                            if name not in velocity:
                                velocity[name] = torch.zeros_like(param.grad)
                            velocity[name].mul_(momentum).add_(-learningRate * param.grad)
                            param.add_(velocity[name])
                            # # Apply Nesterov momentum update
                            # param.add_(momentum * velocity[name] - learningRate * param.grad)

                            #testing lookahead nesterov step
                            param.add_(momentum * velocity[name])

            epochLoss += loss.item()

        epochLoss /= len(trainloader)

        print(f"EPOCH {epoch+1}/{epochs} LOSS: {epochLoss}")

    # 5. Testing the Model
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients during testing
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nAccuracy on test set: {100 * correct / total:.2f}%")

    distribution = get_label_distribution(trainloader)

    # Save the model's state_dict
    torch.save(model.state_dict(), "local_params.pth")
    torch.save(velocity,"local_velocity.pth")
    torch.save(distribution,"local_distribution.pth")
    

if __name__ == "__main__":
   
    model = SimpleNN()  # Initialize the model
    print(model)
    torch.save(model.state_dict(), "global_params.pth")
    velocity = {name: torch.zeros_like(param) for name,param in model.named_parameters()}
   
    for i in range(5):
        train_model(model,velocity)
        model,velocity = send_params()
