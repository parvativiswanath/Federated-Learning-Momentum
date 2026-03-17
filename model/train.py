from copy import deepcopy
import logging
import torch
import torch.nn as nn

from torch import optim

logger = logging.getLogger('fedwan')

##### TRAINING HYPERPARAMETERS #####
EPOCHS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.9
####################################


def train(model, dataset):
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device

    logger.debug("Training:")
    for epoch in range(EPOCHS):

        epochLoss = 0
        for input, target in dataset:
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(input)

            # No need to shape target to one-hot encoding
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epochLoss += loss.item()

        epochLoss /= len(dataset)

        logger.debug(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model

def train_with_momentum(model, dataset, velocity):
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    model.train()
    logger.debug("Training:")
    for epoch in range(EPOCHS):

        epochLoss = 0
        for input, target in dataset:
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

        logger.debug(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model, velocity

def train_with_NAG(model, dataset, velocity):
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    model.train()
    logger.debug("Training:")
    for epoch in range(EPOCHS):

        epochLoss = 0
        for input, target in dataset:
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

        logger.debug(f"EPOCH {epoch} LOSS: {epochLoss}")

    return model, velocity


def train_mime(model, dataset, global_velocity):
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device

    global_model = deepcopy(model)

    model.train()
    logger.debug("Training:")
    for epoch in range(EPOCHS):

        epochLoss = 0
        for input, target in dataset:
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

        logger.debug(f"EPOCH {epoch} LOSS: {epochLoss}")

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



def train_fedprox(model, dataset, global_params, mu=0.01):
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    model.train()
    logger.debug("Training (FedProx):")
    for epoch in range(EPOCHS):
        epochLoss = 0
        for input, target in dataset:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            prox = sum(
                ((p - global_params[n].detach()) ** 2).sum()
                for n, p in model.named_parameters()
            )
            loss = loss + (mu / 2) * prox
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        epochLoss /= len(dataset)
        logger.debug(f"EPOCH {epoch} LOSS: {epochLoss}")
    return model


def train_fednova(model, dataset):
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    model.train()
    logger.debug("Training (FedNova):")
    tau = 0
    for epoch in range(EPOCHS):
        epochLoss = 0
        for input, target in dataset:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
            tau += 1
        epochLoss /= len(dataset)
        logger.debug(f"EPOCH {epoch} LOSS: {epochLoss}")
    return model, tau


def train_scaffold(model, dataset, c_i, c_global):
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    model.train()
    w_start = {name: param.clone().detach() for name, param in model.named_parameters()}
    total_steps = 0
    logger.debug("Training (SCAFFOLD):")
    for epoch in range(EPOCHS):
        epochLoss = 0
        for input, target in dataset:
            input, target = input.to(device), target.to(device)
            model.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        corrected = param.grad + c_global[name] - c_i[name]
                        param.add_(-LEARNING_RATE * corrected)
            epochLoss += loss.item()
            total_steps += 1
        epochLoss /= len(dataset)
        logger.debug(f"EPOCH {epoch} LOSS: {epochLoss}")
    # Option II control variate update: delta_c_i = -c_global + (w_start - w_final) / (K * lr)
    delta_c_i = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            delta_c_i[name] = (
                -c_global[name].clone()
                + (w_start[name] - param.data) / (total_steps * LEARNING_RATE)
            )
    return model, delta_c_i


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
    logger.debug("Testing:")
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