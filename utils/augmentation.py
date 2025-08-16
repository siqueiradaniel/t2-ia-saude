from torchvision import transforms

def get_train_transforms(image_size=224):
    """
    Retorna as transformações de data augmentation para o treino.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size=224):
    """
    Retorna as transformações para validação.
    Apenas redimensiona e normaliza, sem augmentation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms(image_size=224):
    """
    Retorna as transformações para teste.
    Igual à validação.
    """
    return get_val_transforms(image_size=image_size)
