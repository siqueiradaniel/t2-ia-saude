import torch

def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    """
    Função para treinar o modelo. O otimizador é passado como argumento.
    """
    model.to(device)

    # Dicionário para salvar o histórico de perda e acurácia
    history = {
        "train_loss": [],
        "train_acc": []
    }

    print("Iniciando o treinamento...")
    for epoch in range(num_epochs):
        model.train()  # Coloca o modelo em modo de treino
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Loop sobre os lotes de dados
        for inputs, labels, meta_data, uids in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 1. Forward pass: calcula a predição do modelo
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 2. Backward pass e otimização
            optimizer.zero_grad()  # Zera os gradientes da iteração anterior
            loss.backward()        # Calcula os gradientes da perda
            optimizer.step()       # Atualiza os pesos do modelo usando o otimizador (Adam)

            # Coleta de estatísticas
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        # Calcula a perda e acurácia da época
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        print(f"Época {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f} - Acurácia: {epoch_acc:.4f}")

        # Salva o histórico
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc.item())

    print("Treinamento finalizado.")
    return model, history