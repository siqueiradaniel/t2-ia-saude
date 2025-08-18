import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MyCNN, self).__init__()

        # --- Bloco Convolucional 1 ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # << NOVO: Batch Norm após a primeira convolução
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bloco Convolucional 2 ---
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # << NOVO: Batch Norm após a segunda convolução
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bloco Convolucional 3 ---
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # << NOVO: Batch Norm após a terceira convolução
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Bloco Convolucional 4 ---
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Camadas de Classificação (ATUALIZADAS) ---
        # A imagem 224x224 agora é dividida 4 vezes: 224 -> 112 -> 56 -> 28 -> 14
        # A entrada para a camada linear será 256 (canais do conv4) * 14 * 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Passagem pelos blocos 1, 2, 3
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # --- PASSAGEM PELO NOVO BLOCO 4 ---
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten e Classificador (sem alterações)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x