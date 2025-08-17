import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold

# Imports de pastas do projeto
from datasets.dataloader import get_data_loader
from models.my_cnn import MyCNN
from models.pretrained import get_resnet_model
from training.train import train
from training.validate import validate
from training.test import test
from utils.augmentation import get_train_transforms, get_val_transforms
# Removido plot_training_curves pois o histórico é por fold

def load_data(csv_path, dicom_info_filename,
              mass_train_filename, calc_train_filename,
              mass_test_filename, calc_test_filename,
              data_path="./data/"):
    # Sua função load_data (sem alterações)
    dicom_info = pd.read_csv(os.path.join(csv_path, dicom_info_filename))
    mass_train = pd.read_csv(os.path.join(csv_path, mass_train_filename))
    calc_train = pd.read_csv(os.path.join(csv_path, calc_train_filename))
    mass_test = pd.read_csv(os.path.join(csv_path, mass_test_filename))
    calc_test = pd.read_csv(os.path.join(csv_path, calc_test_filename))
    mass_train.rename(columns={"patient_id": "patient id", "breast_density": "breast density"}, inplace=True)
    calc_train.rename(columns={"patient_id": "patient id"}, inplace=True)
    mass_test.rename(columns={"patient_id": "patient id", "breast_density": "breast density"}, inplace=True)
    calc_test.rename(columns={"patient_id": "patient id"}, inplace=True)
    dicom_info = dicom_info[dicom_info["SeriesDescription"] == "cropped images"]
    train_data_csv = pd.concat([mass_train, calc_train], ignore_index=True)
    test_data_csv = pd.concat([mass_test, calc_test], ignore_index=True)
    train_data_csv["SeriesInstanceUID1"] = train_data_csv["image file path"].str.split('/').str[2]
    train_data_csv["SeriesInstanceUID2"] = train_data_csv["cropped image file path"].str.split('/').str[2]
    test_data_csv["SeriesInstanceUID1"] = test_data_csv["image file path"].str.split('/').str[2]
    test_data_csv["SeriesInstanceUID2"] = test_data_csv["cropped image file path"].str.split('/').str[2]
    merge1 = pd.merge(dicom_info, train_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID1", how='inner')
    merge2 = pd.merge(dicom_info, train_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID2", how='inner')
    merge3 = pd.merge(dicom_info, test_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID1", how='inner')
    merge4 = pd.merge(dicom_info, test_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID2", how='inner')
    train_csv_merged = pd.concat([merge1, merge2], ignore_index=True).drop_duplicates()
    test_csv_merged = pd.concat([merge3, merge4], ignore_index=True).drop_duplicates()
    return train_csv_merged, test_csv_merged

def main():
    # --- Configurações ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32
    data_path = "./data/"
    csv_path = "./data/csv/"
    
    # --- Escolha do Modelo ---
    USE_RESNET = False  # Mude para False para usar sua MyCNN
    
    # --- Configurações da Validação Cruzada ---
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []

    # --- Carregar Dados ---
    print("Carregando e preparando os dados...")
    train_csv_full, test_csv = load_data(
        csv_path=csv_path,
        dicom_info_filename="dicom_info.csv",
        mass_train_filename="mass_case_description_train_set.csv",
        calc_train_filename="calc_case_description_train_set.csv",
        mass_test_filename="mass_case_description_test_set.csv",
        calc_test_filename="calc_case_description_test_set.csv",
        data_path=data_path
    )
    
    # --- Loop de Validação Cruzada ---
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_csv_full)):
        print(f"\n========== FOLD {fold + 1}/{N_SPLITS} ==========")
        train_df_fold = train_csv_full.iloc[train_indices]
        val_df_fold = train_csv_full.iloc[val_indices]

        train_imgs_path = [data_path + path for path in train_df_fold['image_path']]
        train_labels = list(train_df_fold['pathology'])
        val_imgs_path = [data_path + path for path in val_df_fold['image_path']]
        val_labels = list(val_df_fold['pathology'])

        train_loader = get_data_loader(
            imgs_path=train_imgs_path, labels=train_labels,
            transform=get_train_transforms(), batch_size=batch_size, shuf=True
        )
        val_loader = get_data_loader(
            imgs_path=val_imgs_path, labels=val_labels,
            transform=get_val_transforms(), batch_size=batch_size, shuf=False
        )

        if USE_RESNET:
            model = get_resnet_model(num_classes=2, pretrained=True)
        else:
            model = MyCNN(num_classes=2)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        trained_model, _ = train(
            model=model, dataloader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device, num_epochs=num_epochs
        )
        
        val_loss, val_acc = validate(trained_model, val_loader, criterion, device)
        fold_results.append(val_acc.item())
        print(f"Resultado do Fold {fold + 1}: Acurácia de Validação = {val_acc.item():.4f}")

    # --- Análise dos Resultados da Validação Cruzada ---
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print("\n\n========== RESULTADO DA VALIDAÇÃO CRUZADA ==========")
    print(f"Modelo: {'ResNet50' if USE_RESNET else 'MyCNN'}")
    print(f"Acurácia Média nos {N_SPLITS} folds: {mean_acc:.4f}")
    print(f"Desvio Padrão da Acurácia: {std_acc:.4f}")
    print(f"Resultados individuais por fold: {[round(r, 4) for r in fold_results]}")
    
    # --- Treinamento Final e Avaliação no Conjunto de Teste ---
    print("\n\n========== TREINAMENTO FINAL COM DADOS COMPLETOS ==========")
    # Recarrega o DataLoader com todos os dados de treino
    full_train_loader = get_data_loader(
        imgs_path=[data_path + path for path in train_csv_full['image_path']],
        labels=list(train_csv_full['pathology']),
        transform=get_train_transforms(), batch_size=batch_size, shuf=True
    )
    
    # Re-inicializa o modelo final
    if USE_RESNET:
        final_model = get_resnet_model(num_classes=2, pretrained=True)
    else:
        final_model = MyCNN(num_classes=2)
    final_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

    # Treina com todos os dados
    final_trained_model, _ = train(
        model=final_model, dataloader=full_train_loader, criterion=criterion,
        optimizer=optimizer, device=device, num_epochs=num_epochs
    )
    
    print("\n========== AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ==========")
    test_imgs_path = [data_path + path for path in test_csv['image_path']]
    test_labels = list(test_csv['pathology'])
    test_loader = get_data_loader(
        imgs_path=test_imgs_path, labels=test_labels,
        transform=get_val_transforms(), batch_size=batch_size, shuf=False
    )
    
    test(final_trained_model, test_loader, criterion, device)

    # Salva o modelo final
    model_name = "resnet50_final.pth" if USE_RESNET else "my_cnn_final.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(final_trained_model.state_dict(), f"models/{model_name}")
    print(f"\n✅ Modelo final treinado e salvo em models/{model_name}")

if __name__ == "__main__":
    main()