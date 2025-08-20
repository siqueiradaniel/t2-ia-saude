import torch
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def test(model, dataloader, criterion, device):
    """
    Função para testar o modelo e calcular métricas detalhadas de performance.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []  # Probabilidade da classe positiva

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            if isinstance(batch, (tuple, list)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs, labels = batch["inputs"], batch["labels"]

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    test_loss = running_loss / max(1, total_samples)
    test_acc = running_corrects / max(1, total_samples)

    all_preds = torch.cat(all_preds) if len(all_preds) else torch.tensor([])
    all_labels = torch.cat(all_labels) if len(all_labels) else torch.tensor([])
    all_probs = torch.cat(all_probs) if len(all_probs) else torch.tensor([])

    print("--- Relatório de Classificação ---")
    target_names = ['Classe Negativa', 'Classe Positiva']
    report_dict = {}
    if all_labels.numel() > 0:
        report_str = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=target_names)
        report_dict = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=target_names, output_dict=True)
        print(report_str)
    else:
        print("Sem amostras para relatório.")
    print("------------------------------------")

    roc_auc = None
    if all_labels.numel() > 0:
        fpr, tpr, thresholds = roc_curve(all_labels.numpy(), all_probs.numpy())
        roc_auc = auc(fpr, tpr)
        print(f"AUC (Area Under Curve): {roc_auc:.4f}\n")
        # Plot opcional (comente se rodar em ambiente sem display)
        try:
            plt.figure()
            plt.plot(fpr, tpr, lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taxa de Falsos Positivos (FPR)')
            plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
            plt.title('Curva ROC')
            plt.legend(loc="lower right")
            plt.show()
        except Exception:
            pass

    print(f"Resultado do Teste -> Loss: {test_loss:.4f} - Acurácia: {test_acc:.4f}")
    return test_loss, test_acc, report_dict, roc_auc
