import torch
from torch.nn.utils import clip_grad_norm_
from training.validate import validate 

def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    val_loader=None,
    scheduler=None,
    num_epochs=10,
    grad_clip_norm=None,
    accum_steps=1,
    use_amp=None
):
    """
    Função de treino refatorada que integra a validação por época e o scheduler.
    """
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if use_amp is None:
        use_amp = torch.cuda.is_available() and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("Iniciando o treinamento...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if batch is None: continue
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accum_steps
            
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                running_loss += (loss.detach() * accum_steps).item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += labels.size(0)

        epoch_loss = running_loss / max(1, total_samples)
        epoch_acc = running_corrects / max(1, total_samples)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        log_message = f"Época {epoch+1}/{num_epochs} -> Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f}"

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            if scheduler:
                scheduler.step(val_loss)
            
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc.item())
            log_message += f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc.item():.4f}"
        
        print(log_message)

    print("Treinamento finalizado.")
    return model, history