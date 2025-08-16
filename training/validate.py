import torch

def validate(model, dataloader, criterion, device):
    model.eval()  # modo avaliação
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():  # sem gradientes
        for inputs, labels, meta_data, uids in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Estatísticas
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    val_loss = running_loss / total_samples
    val_acc = running_corrects.double() / total_samples

    print(f"Validation - Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")
    return val_loss, val_acc
