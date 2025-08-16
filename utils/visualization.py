import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues', normalize='true')
    plt.title("Matriz de Confusão")
    plt.show()

def plot_roc_curve(y_true, y_prob):
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve")
    plt.show()

def show_images_with_predictions(images, true_labels, pred_labels, class_names=None, n=5):
    """
    images: tensor ou numpy array (B, C, H, W)
    true_labels: lista ou tensor
    pred_labels: lista ou tensor
    n: número de imagens para mostrar
    """
    import numpy as np
    import torchvision

    images = images[:n]
    true_labels = true_labels[:n]
    pred_labels = pred_labels[:n]

    plt.figure(figsize=(15, 5))
    for i in range(n):
        img = images[i].permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        title = f"True: {true_labels[i]}\nPred: {pred_labels[i]}"
        if class_names:
            title = f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}"
        plt.title(title)
        plt.axis('off')
    plt.show()

def plot_training_curves(history, save_path="training_curves.png"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    import matplotlib.pyplot as plt
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
