import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MyCNN, self).__init__()

        # Camada 1: Convolução + ReLU + Pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Camada 2: Convolução + ReLU + Pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Camada 3: Convolução + ReLU + Pooling
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten será feito no forward
        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 224x224 -> /2 -> 112 -> /2 -> 56 -> /2 -> 28
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout para regularização
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Passagem pelas camadas convolucionais
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Camadas totalmente conectadas
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
