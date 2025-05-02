import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from collections import Counter
from tqdm import tqdm
from adan import Adan

# ---------------------------
# 1. Load and preprocess data
# ---------------------------
data = pd.read_csv('paddy_field_client1.csv')
test_data = pd.read_csv('paddyfield_test.csv')

required_cols = ['Water Level', 'Moisture Percentage', 'Light Intensity', 'Temperature', 'Humidity', 'Cluster']
assert set(required_cols).issubset(data.columns), "Missing required columns in training set"
assert set(required_cols).issubset(test_data.columns), "Missing required columns in test set"

features_train = data[required_cols[:-1]].values
labels_train = data['Cluster'].values
features_test = test_data[required_cols[:-1]].values
labels_test = test_data['Cluster'].values

# Normalize
mean = np.mean(features_train, axis=0)
std = np.std(features_train, axis=0)
features_train = (features_train - mean) / std
features_test = (features_test - mean) / std

# Convert to tensors
features_train_tensor = torch.tensor(features_train, dtype=torch.float32)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)
features_test_tensor = torch.tensor(features_test, dtype=torch.float32)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

train_dataset = TensorDataset(features_train_tensor, labels_train_tensor)
test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------------
# 2. Neural Network Definition
# ---------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)  # NLLLoss-compatible output
        return x


# ---------------------------
# 5. Training Function with Adan
# ---------------------------
def train_with_adan(model, dataset, cli_velocity):
    epochs = 2
    learningRate = 0.001
    betas = (0.98, 0.92, 0.99)
    weight_decay = 0.02
    eps = 1e-8

    optimizer = Adan(model.parameters(), lr=learningRate, betas=betas,
                     weight_decay=weight_decay, eps=eps, model=model)
    optimizer.set_velocity_terms(cli_velocity)
    criterion = nn.NLLLoss()

    print("Training:")
    for epoch in range(epochs):
        epochLoss = 0
        for input, target in tqdm(dataset):
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        epochLoss /= len(dataset)
        print(f"EPOCH {epoch+1} LOSS: {epochLoss}")

    # Evaluate model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nAccuracy on test set: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "local_params.pth")
    torch.save(optimizer.get_velocity_terms(), "local_velocity.pth")


    return model, optimizer.get_velocity_terms()

# ---------------------------
# 6. Server Communication
# ---------------------------
def send_params():
    url = 'http://10.67.67.253:5000/upload'
        
    # Paths to the files you want to upload
    file1_path = 'local_params.pth'
    file2_path = 'local_velocity.pth'

    # Open the files in binary mode
    with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
        # Prepare the files for upload
        files = {
            'file1': file1,
            'file2': file2,
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

if __name__ == "__main__":
   
    model = SimpleNN()  # Initialize the model
    print(model)
    torch.save(model.state_dict(), "global_params.pth")
    velocity = {}
    for name, param in model.named_parameters():
        velocity[name] = {
            'exp_avg': torch.zeros_like(param),
            'exp_avg_diff': torch.zeros_like(param),
            'exp_avg_sq': torch.zeros_like(param)
        }
   
    for i in range(5):
        train_with_adan(model,trainloader, velocity)
        model,velocity = send_params()