import torch
from torch.nn.utils import clip_grad_norm_

def train(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    grad_clip_norm=None,          # e.g., 1.0 to enable clipping (L2 norm)
    accum_steps=1,                # >1 to enable gradient accumulation
    use_amp=None                  # None -> auto (True if CUDA available), or force True/False
):
    """
    Treina o modelo com opções de:
      - Clipping de gradiente (grad_clip_norm)
      - Acumulação de gradientes (accum_steps)
      - AMP (mixed precision) com GradScaler (use_amp)
    """
    model.to(device)
    history = {"train_loss": [], "train_acc": []}

    # Decisão automática do AMP
    if use_amp is None:
        use_amp = torch.cuda.is_available() and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print("Iniciando o treinamento...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader):
            if batch is None:
                continue

            # Suporta (inputs, labels, meta_data, uids)
            if isinstance(batch, (tuple, list)):
                inputs, labels = batch[0], batch[1]
            else:
                # caso o dataloader retorne um dict
                inputs, labels = batch["inputs"], batch["labels"]

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / max(1, accum_steps)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels) / max(1, accum_steps)

            # backward (com AMP ou não)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Atualização a cada accum_steps
            do_step = ((step + 1) % max(1, accum_steps) == 0)

            if do_step:
                # Grad clipping (antes do step)
                if grad_clip_norm is not None:
                    if scaler is not None:
                        # Unscale para clipping correto
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Estatísticas (loss antes da divisão por accum_steps)
            with torch.no_grad():
                running_loss += (loss.detach() * max(1, accum_steps)).item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += labels.size(0)

        epoch_loss = running_loss / max(1, total_samples)
        epoch_acc = running_corrects / max(1, total_samples)

        print(f"Época {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f} - Acurácia: {epoch_acc:.4f}")
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

    print("Treinamento finalizado.")
    return model, history
