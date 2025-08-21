# Conteúdo para models/pretrained.py
import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes=2, pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Congela todos os parâmetros primeiro
    for param in model.parameters():
        param.requires_grad = False

    # Descongela os parâmetros do último bloco convolucional (layer4) e da camada FC
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # A nova camada FC já vem com requires_grad=True por padrão

    return model