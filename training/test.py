import torch

def test(model, dataloader, criterion, device):
    model.eval()  # modo avaliação
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
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

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    test_loss = running_loss / total_samples
    test_acc = running_corrects.double() / total_samples

    # Concatenar todas as predições e labels para métricas adicionais
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print(f"Test - Loss: {test_loss:.4f} - Acc: {test_acc:.4f}")
    return test_loss, test_acc, all_preds, all_labels
