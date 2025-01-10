import torch.nn as nn
import torch.nn.functional as F

inputdim = 28     #MNIST
#inputdim = 32     #CIFAR-10

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5)  # Adjust input channels to 3
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(64 * 5 * 5, 256)  # Adjust for CIFAR-10 dimensions
#         self.fc2 = nn.Linear(256, 10)  # CIFAR-10 has 10 classes

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 64 * 5 * 5)  # Adjust for new feature map size
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5)  # Adjust input channels to 3
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(64 * 5 * 5, 256)  # Adjust for CIFAR-10 dimensions
#         self.fc2 = nn.Linear(256, 10)  # CIFAR-10 has 10 classes

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 64 * 5 * 5)  # Adjust for new feature map size
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# class DNN(nn.Module):
#     def __init__(self):
#         super(DNN, self).__init__()
#         self.fc1 = nn.Linear(inputdim*inputdim*3, 64)
#         self.fc2 = nn.Linear(64, 32)
#         #self.fc3 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 10)

#     def forward(self, x):
#         x = x.view(-1, inputdim*inputdim*3)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         #x = F.relu(self.fc3(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
    
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)