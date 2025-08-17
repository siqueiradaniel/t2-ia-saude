import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Imports do projeto
from datasets.dataloader import get_data_loader
from models.my_cnn import MyCNN
from training.train import train
from training.validate import validate
from training.test import test
from utils.augmentation import get_train_transforms, get_val_transforms
from utils.visualization import plot_training_curves


def load_data(csv_path, dicom_info_filename,
              mass_train_filename, calc_train_filename,
              mass_test_filename, calc_test_filename,
              data_path="./data/"):
    """
    Carrega e organiza os CSVs e retorna DataFrames de treino e teste.
    """
    dicom_info = pd.read_csv(os.path.join(csv_path, dicom_info_filename))
    mass_train = pd.read_csv(os.path.join(csv_path, mass_train_filename))
    calc_train = pd.read_csv(os.path.join(csv_path, calc_train_filename))
    mass_test = pd.read_csv(os.path.join(csv_path, mass_test_filename))
    calc_test = pd.read_csv(os.path.join(csv_path, calc_test_filename))

    # Renomear colunas
    mass_train.rename(columns={"patient_id": "patient id", "breast_density": "breast density"}, inplace=True)
    calc_train.rename(columns={"patient_id": "patient id"}, inplace=True)
    mass_test.rename(columns={"patient_id": "patient id", "breast_density": "breast density"}, inplace=True)
    calc_test.rename(columns={"patient_id": "patient id"}, inplace=True)

    # Filtrar imagens cropped
    dicom_info = dicom_info[dicom_info["SeriesDescription"] == "cropped images"]

    # Concatenar
    train_data_csv = pd.concat([mass_train, calc_train], ignore_index=True)
    test_data_csv = pd.concat([mass_test, calc_test], ignore_index=True)

    # Criar colunas auxiliares para merge
    train_data_csv["SeriesInstanceUID1"] = train_data_csv["image file path"].str.split('/').str[2]
    train_data_csv["SeriesInstanceUID2"] = train_data_csv["cropped image file path"].str.split('/').str[2]
    test_data_csv["SeriesInstanceUID1"] = test_data_csv["image file path"].str.split('/').str[2]
    test_data_csv["SeriesInstanceUID2"] = test_data_csv["cropped image file path"].str.split('/').str[2]

    # Merge
    merge1 = pd.merge(dicom_info, train_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID1", how='inner')
    merge2 = pd.merge(dicom_info, train_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID2", how='inner')
    merge3 = pd.merge(dicom_info, test_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID1", how='inner')
    merge4 = pd.merge(dicom_info, test_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID2", how='inner')

    train_csv_merged = pd.concat([merge1, merge2], ignore_index=True).drop_duplicates()
    test_csv_merged = pd.concat([merge3, merge4], ignore_index=True).drop_duplicates()

    return train_csv_merged, test_csv_merged


def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 30
    learning_rate = 0.001
    num_classes = 2
    batch_size = 32

    data_path = "./data/"
    csv_path = "./data/csv/"

    # Carregar dados
    train_csv, test_csv = load_data(
        csv_path=csv_path,
        dicom_info_filename="dicom_info.csv",
        mass_train_filename="mass_case_description_train_set.csv",
        calc_train_filename="calc_case_description_train_set.csv",
        mass_test_filename="mass_case_description_test_set.csv",
        calc_test_filename="calc_case_description_test_set.csv",
        data_path=data_path
    )

    # Criar caminhos e labels
    train_imgs_path = [data_path + path for path in train_csv['image_path']]
    train_labels = list(train_csv['pathology'])

    test_imgs_path = [data_path + path for path in test_csv['image_path']]
    test_labels = list(test_csv['pathology'])

    # DataLoaders
    train_loader = get_data_loader(
        imgs_path=train_imgs_path,
        labels=train_labels,
        transform=get_train_transforms(),
        batch_size=batch_size,
        shuf=True
    )

    val_loader = get_data_loader(
        imgs_path=train_imgs_path,   # <- ideal seria split train/val
        labels=train_labels,
        transform=get_val_transforms(),
        batch_size=batch_size,
        shuf=False
    )

    test_loader = get_data_loader(
        imgs_path=test_imgs_path,
        labels=test_labels,
        transform=get_val_transforms(),
        batch_size=batch_size,
        shuf=False
    )

    # Modelo, Loss e Otimizador
    model = MyCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treinamento
    trained_model, history = train(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )

    # Validação
    val_loss, val_acc = validate(trained_model, val_loader, criterion, device)
    print(f"✅ Validação - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Teste
    test_loss, test_acc, all_labels, all_preds = test(trained_model, test_loader, criterion, device)
    print(f"✅ Teste - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Curvas de treino
    plot_training_curves(history, save_path="results/training_curves.png")

    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    torch.save(trained_model.state_dict(), "models/my_cnn.pth")
    print("✅ Modelo treinado e salvo em models/my_cnn.pth")


if __name__ == "__main__":
    main()
