from copy import deepcopy
import torch
import torch.nn as nn

from torch import optim

from tqdm import tqdm

##### TRAINING HYPERPARAMETERS #####
EPOCHS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.9
####################################


def train(model, dataset):
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device

    print("Training:")
    for epoch in range(EPOCHS):

        epochLoss = 0
        for input, target in tqdm(dataset):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(input)

            # No need to shape target to one-hot encoding
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model

def train_with_momentum(model, dataset, velocity):
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    model.train()
    print("Training:")
    for epoch in range(EPOCHS):

        epochLoss = 0
        for input, target in tqdm(dataset):
            input, target = input.to(device), target.to(device)

            model.zero_grad()

            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # velocity[name] = MOMENTUM * velocity[name] + (1-MOMENTUM)*param.grad    #correct equation
                        velocity[name] = MOMENTUM * velocity[name] + param.grad      #pytorch implementation
                        param.add_(-(LEARNING_RATE * velocity[name].detach_()))

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model, velocity

def train_with_NAG(model, dataset, velocity):
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    model.train()
    print("Training:")
    for epoch in range(EPOCHS):

        epochLoss = 0
        for input, target in tqdm(dataset):
            input, target = input.to(device), target.to(device)

            model.zero_grad()

            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Update velocity using momentum
                        if name not in velocity:
                            velocity[name] = torch.zeros_like(param.grad)
                        velocity[name].mul_(MOMENTUM).add_(-LEARNING_RATE * param.grad)
                        param.add_(velocity[name])
                        # Apply Nesterov lookahead step
                        param.add_(MOMENTUM * velocity[name])

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model, velocity


def train_mime(model, dataset, global_velocity):
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device

    global_model = deepcopy(model)

    model.train()
    print("Training:")
    for epoch in range(EPOCHS):

        epochLoss = 0
        for input, target in tqdm(dataset):
            input, target = input.to(device), target.to(device)

            model.zero_grad()

            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.add_(-LEARNING_RATE * (MOMENTUM * global_velocity[name].detach_() + param.grad))

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        print(f"EPOCH {epoch} LOSS: {epochLoss}")

    #Compute full batch gradient based on server parameters
    data, target = next(iter(dataset))
    data, target = data.to(device), target.to(device)
    global_model.zero_grad()
    output = global_model(data)
    loss = criterion(output, target)
    loss.backward()
    gradients = {}
    for name, param in global_model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
    
    return model, gradients



#ORIGINAL TESTING CODE
# def test(model, testSet):
#     print("Testing:")
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for input, target in testSet:
#             output = model(input)
#             correct += (output.argmax(1) == target).sum().item()
#             total += target.size(0)
#     return correct / total

def test(model,testSet):
    print("Testing:")
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for input, target in testSet:
            input, target = input.to(device), target.to(device)
            output = model(input)  # Forward pass
            loss = criterion(output, target)  # Compute loss

            total_loss += loss.item()  # Accumulate total loss
            correct += (output.argmax(1) == target).sum().item()  # Count correct predictions
            total += target.size(0)  # Total number of samples

    # Calculate metrics
    accuracy = correct / total
    average_loss = total_loss / len(testSet)  # Average loss per batch
    return accuracy, average_loss