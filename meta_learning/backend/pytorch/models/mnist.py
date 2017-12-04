import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class LeNet(nn.Module):
#     def __init__(self, n_classes):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 5, 5, 1)
#         self.conv2 = nn.Conv2d(5, 10, 5, 1)
#         self.fc1 = nn.Linear(4*4*10, 300)
#         self.fc2 = nn.Linear(300, n_classes) 

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(x)

#         x = x.view(-1, 4*4*10)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

class LeNet(nn.Module):
    def __init__(self, n_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
