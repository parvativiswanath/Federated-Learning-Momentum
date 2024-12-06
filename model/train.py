import torch
import torch.nn as nn

from torch import optim

from tqdm import tqdm


def train(model, dataset):

    ##### TRAINING HYPERPARAMETERS #####
    epochs = 3
    learningRate = 0.01
    momentum = 0.9
    #optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    criterion = nn.NLLLoss()
    ####################################

    print("Training:")
    for epoch in range(epochs):

        epochLoss = 0
        for input, target in tqdm(dataset):

            # Reset such that only gradients that pertain
            # to the current input are used
            optimizer.zero_grad()

            # Forward
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
    ##### TRAINING HYPERPARAMETERS #####
    epochs = 3
    learningRate = 0.001
    momentum = 0.9
    #optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    criterion = nn.NLLLoss()
    ####################################

    model.train()
    print("Training:")
    for epoch in range(epochs):

        epochLoss = 0
        for input, target in tqdm(dataset):

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
                        # velocity[name] = momentum * velocity[name] + (1-momentum)*param.grad    #correct equation
                        velocity[name] = momentum * velocity[name] + param.grad      #pytorch implementation
                        param.add_(-(learningRate * velocity[name].detach_()))

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model, velocity

def train_with_NAG(model, dataset, velocity):
    ##### TRAINING HYPERPARAMETERS #####
    epochs = 3
    learningRate = 0.001
    momentum = 0.9
    #optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    criterion = nn.NLLLoss()
    ####################################

    model.train()
    print("Training:")
    for epoch in range(epochs):

        epochLoss = 0
        for input, target in tqdm(dataset):

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

        epochLoss /= len(dataset)

        print(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model, velocity

        # for i, (velocity, parameter, gradient) in enumerate(zip(self.velocity, parameters, gradients)):
        #     # Update velocity: velocity = momentum * velocity - lr * gradient
        #     velocity = self.momentum * velocity - self.lr * gradient

        #     # Update parameter using Nesterov update: parameter += velocity
        #     parameter += velocity

        #     # Update attributes
        #     self.velocity[i] = velocity


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
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for input, target in testSet:
            output = model(input)  # Forward pass
            loss = criterion(output, target)  # Compute loss

            total_loss += loss.item()  # Accumulate total loss
            correct += (output.argmax(1) == target).sum().item()  # Count correct predictions
            total += target.size(0)  # Total number of samples

    # Calculate metrics
    accuracy = correct / total
    average_loss = total_loss / len(testSet)  # Average loss per batch
    return accuracy, average_loss