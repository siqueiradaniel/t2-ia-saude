import torch

def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": []
    }

    for epoch in range(num_epochs):
        model.train()  # modo treino
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels, meta_data, uids in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Estatísticas
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

        # Salva histórico
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc.item())

    return model, history
