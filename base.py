import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from collections import Counter

# 2. Defining a Simple Neural Network (Feedforward)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)  # Input layer: 28x28 pixels -> 128 neurons
        self.fc2 = nn.Linear(64, 32)       # Hidden layer: 128 -> 64 neurons
        self.fc3 = nn.Linear(32, 10)        # Output layer: 64 -> 10 neurons (for 10 classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten 28x28 images to a 1D vector
        x = torch.relu(self.fc1(x))  # ReLU activation for first hidden layer
        x = torch.relu(self.fc2(x))  # ReLU activation for second hidden layer
        x = self.fc3(x)             # Output layer (no activation, raw logits)
        return x

def train_model():

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

def base():
    model = SimpleNN()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Downloading MNIST dataset (train and test)
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    model = train(model, trainLoader)
    testLoss = test(model, testLoader)

    print("Test accuracy: %.2f%%" % (testLoss * 100))


if __name__ == "__main__":
    base()
