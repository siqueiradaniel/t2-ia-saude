import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes=2, pretrained=True):
    """
    Carrega um modelo ResNet50 pré-treinado e o adapta para a classificação,
    substituindo a última camada para ter 'num_classes' saídas.

    Args:
        num_classes (int): O número de classes de saída. Para o seu problema, é 2.
        pretrained (bool): Se True, carrega os pesos pré-treinados no ImageNet.

    Returns:
        torch.nn.Module: O modelo ResNet50 modificado.
    """
    # 1. Carrega a arquitetura ResNet50 com os pesos mais recentes do ImageNet
    # Usar a API 'weights' é a prática moderna em vez de 'pretrained=True'
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # 2. Congela os pesos de todas as camadas convolucionais
    # Isso impede que o conhecimento aprendido no ImageNet seja perdido durante o treino.
    # Apenas a nova camada de classificação será treinada.
    for param in model.parameters():
        param.requires_grad = False

    # 3. Substitui a última camada (o "classifier head")
    # Obtém o número de características de entrada da camada fully connected (fc) original
    num_ftrs = model.fc.in_features

    # Cria uma nova camada linear que recebe 'num_ftrs' e produz 'num_classes' saídas (logits).
    # Esta nova camada terá `requires_grad=True` por padrão.
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model