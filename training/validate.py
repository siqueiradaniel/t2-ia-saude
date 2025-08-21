import torch

def validate(model, dataloader, criterion, device):
    """
    Função limpa que apenas executa a validação e retorna as métricas.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            # Garante que o batch seja desempacotado corretamente
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                continue # Pula batches malformados

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += labels.size(0)

    val_loss = running_loss / max(1, total_samples)
    val_acc = torch.tensor(running_corrects / max(1, total_samples), dtype=torch.float32)

    return val_loss, val_acc