import torch
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def test(model, dataloader, criterion, device):
    """
    Função para testar o modelo e calcular métricas detalhadas de performance.
    """
    model.eval()  # Modo avaliação
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = [] # Armazena as probabilidades para a curva ROC

    with torch.no_grad():
        for inputs, labels, meta_data, uids in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Probabilidades para a curva ROC
            # Acessa a saída da classe positiva (classe 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            # Predições para acurácia e outras métricas
            _, preds = torch.max(outputs, 1)

            # Estatísticas
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    test_loss = running_loss / total_samples
    test_acc = running_corrects.double() / total_samples

    # Concatenar todos os tensores da lista
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # --- SEÇÃO DE MÉTRICAS DETALHADAS ---

    # 1. Relatório de Classificação (Precision, Recall, F1-score)
    print("--- Relatório de Classificação ---")
    # Altere target_names se tiver nomes específicos para as classes
    target_names = ['Classe Negativa', 'Classe Positiva']
    report_str = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=target_names)
    report_dict = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=target_names, output_dict=True)
    print(report_str)
    print("------------------------------------")

    # 2. Curva ROC e AUC
    fpr, tpr, thresholds = roc_curve(all_labels.numpy(), all_probs.numpy())
    roc_auc = auc(fpr, tpr)
    print(f"AUC (Area Under Curve): {roc_auc:.4f}\n")

    # Plotar a curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()


    print(f"Resultado do Teste -> Loss: {test_loss:.4f} - Acurácia: {test_acc:.4f}")

    return test_loss, test_acc, report_dict, roc_auc