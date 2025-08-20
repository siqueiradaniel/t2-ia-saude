from torchvision import transforms

def get_train_transforms(image_size=224):
    """
    Transformações de data augmentation otimizadas para imagens de mamografia.
    """
    return transforms.Compose([
        # Redimensiona a imagem para o tamanho de entrada do modelo
        transforms.Resize((image_size, image_size)),

        # Adiciona ruído suave para simular variações de captura
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),

        # Inversão horizontal (comum em mamografia)
        transforms.RandomHorizontalFlip(p=0.5),

        # Inversão vertical para acomodar diferentes orientações de imagem
        transforms.RandomVerticalFlip(p=0.5),
        
        # Ajusta contraste e brilho para simular variações de iluminação e densidade
        # Os valores são menores para não criar imagens clinicamente irreais
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        
        # Rotaciona a imagem em pequenos ângulos
        transforms.RandomRotation(degrees=180),

        # Converte a imagem para tensor
        transforms.ToTensor(),

        # Normalização para padronizar os dados de entrada
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
