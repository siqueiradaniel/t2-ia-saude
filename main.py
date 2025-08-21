import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon, ttest_rel

# Imports de pastas do projeto
from datasets.dataloader import get_data_loader
from models.my_cnn import MyCNN
from models.pretrained import get_resnet_model
from training.train import train
from training.test import test
from utils.augmentation import get_train_transforms, get_val_transforms


def load_data(csv_path, dicom_info_filename,
              mass_train_filename, calc_train_filename,
              mass_test_filename, calc_test_filename,
              data_path="./data/"):
    # (Esta função está correta, mantida como original)
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
    train_csv_merged['image_path'] = train_csv_merged['image_path'].str.replace('CBIS-DDSM/', '', regex=False)
    test_csv_merged['image_path'] = test_csv_merged['image_path'].str.replace('CBIS-DDSM/', '', regex=False)
    return train_csv_merged, test_csv_merged


def main():
    # --- Configurações ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    num_epochs = 10
    learning_rate = 0.0001
    batch_size = 32
    data_path = "./data/"
    csv_path = "./data/csv/"
    ISDEVELOPING = False
    RUN_RESNET = False

    GRAD_CLIP_NORM = 1.0
    ACCUM_STEPS = 1
    USE_AMP = None

    N_SPLITS = 2
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    print("Carregando e preparando os dados...")
    train_csv_full, test_csv = load_data(
        csv_path=csv_path, dicom_info_filename="dicom_info.csv",
        mass_train_filename="mass_case_description_train_set.csv", calc_train_filename="calc_case_description_train_set.csv",
        mass_test_filename="mass_case_description_test_set.csv", calc_test_filename="calc_case_description_test_set.csv",
        data_path=data_path
    )
    
    patient_info = train_csv_full.groupby('patient id')['pathology'].max().reset_index()
    
    if (ISDEVELOPING):
        print ("\n----------------- ATENÇÃO: MODO DE DESENVOLVIMENTO ATIVADO -----------------\n")
        DEV_FRAC = 0.05
        temp_df_sampled = patient_info.sample(frac=DEV_FRAC, random_state=42)
        train_csv_full = train_csv_full[train_csv_full['patient id'].isin(temp_df_sampled['patient id'])].reset_index(drop=True)
        test_csv = test_csv.sample(frac=DEV_FRAC, random_state=42).reset_index(drop=True)
        patient_info = train_csv_full.groupby('patient id')['pathology'].max().reset_index()

    patient_ids = patient_info['patient id']
    patient_labels = patient_info['pathology']

    pathology_counts = train_csv_full['pathology'].value_counts()
    count_class_0 = pathology_counts.get(0, 1)
    count_class_1 = pathology_counts.get(1, 1)
    class_weights = torch.tensor([(count_class_0 + count_class_1) / count_class_0, (count_class_0 + count_class_1) / count_class_1], dtype=torch.float32).to(device)
    print(f"Pesos das classes calculados: {class_weights}")

    # --- Validação Cruzada do MyCNN ---
    # print("\n\n========== VALIDANDO MyCNN COM VALIDAÇÃO CRUZADA ==========")
    # mycnn_fold_results = []
    # for fold, (train_indices, val_indices) in enumerate(kf.split(patient_ids, y=patient_labels)):
    #     print(f"\n========== FOLD {fold + 1}/{N_SPLITS} ==========")
    #     train_patient_ids = patient_ids.iloc[train_indices]
    #     val_patient_ids = patient_ids.iloc[val_indices]
    #     train_df_fold = train_csv_full[train_csv_full['patient id'].isin(train_patient_ids)]
    #     val_df_fold = train_csv_full[train_csv_full['patient id'].isin(val_patient_ids)]
        
    #     train_loader = get_data_loader(imgs_path=[data_path + path for path in train_df_fold['image_path']], labels=list(train_df_fold['pathology']), transform=get_train_transforms(), batch_size=batch_size, shuf=True)
    #     val_loader = get_data_loader(imgs_path=[data_path + path for path in val_df_fold['image_path']], labels=list(val_df_fold['pathology']), transform=get_val_transforms(), batch_size=batch_size, shuf=False)

    #     model = MyCNN(num_classes=2).to(device)
    #     criterion = nn.CrossEntropyLoss(weight=class_weights)
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #     # CORREÇÃO: Removido o argumento 'verbose=True'
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    #     _, history = train(
    #         model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer,
    #         scheduler=scheduler, device=device, num_epochs=num_epochs, grad_clip_norm=GRAD_CLIP_NORM,
    #         accum_steps=ACCUM_STEPS, use_amp=USE_AMP
    #     )
        
    #     final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
    #     mycnn_fold_results.append(final_val_acc)
    #     print(f"Resultado do Fold {fold + 1}: Acurácia de Validação Final = {final_val_acc:.4f}")

    if RUN_RESNET:
        print("\n\n========== VALIDANDO ResNet50 COM VALIDAÇÃO CRUZADA ==========")
        resnet_fold_results = []
        for fold, (train_indices, val_indices) in enumerate(kf.split(patient_ids, y=patient_labels)):
            print(f"\n========== FOLD {fold + 1}/{N_SPLITS} ==========")
            train_patient_ids = patient_ids.iloc[train_indices]
            val_patient_ids = patient_ids.iloc[val_indices]
            train_df_fold = train_csv_full[train_csv_full['patient id'].isin(train_patient_ids)]
            val_df_fold = train_csv_full[train_csv_full['patient id'].isin(val_patient_ids)]
            
            train_loader = get_data_loader(imgs_path=[data_path + path for path in train_df_fold['image_path']], labels=list(train_df_fold['pathology']), transform=get_train_transforms(), batch_size=batch_size, shuf=True)
            val_loader = get_data_loader(imgs_path=[data_path + path for path in val_df_fold['image_path']], labels=list(val_df_fold['pathology']), transform=get_val_transforms(), batch_size=batch_size, shuf=False)

            model = get_resnet_model(num_classes=2, pretrained=True).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # CORREÇÃO: Removido o argumento 'verbose=True'
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

            _, history = train(
                model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer,
                scheduler=scheduler, device=device, num_epochs=num_epochs, grad_clip_norm=GRAD_CLIP_NORM,
                accum_steps=ACCUM_STEPS, use_amp=USE_AMP
            )
            
            final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
            resnet_fold_results.append(final_val_acc)
            print(f"Resultado do Fold {fold + 1}: Acurácia de Validação Final = {final_val_acc:.4f}")

        print("\n\n========== TESTE ESTATÍSTICO ==========")
        mc = np.asarray(mycnn_fold_results); rn = np.asarray(resnet_fold_results)
        print(f"Resultados de Acurácia do MyCNN: {mc.tolist()}"); print(f"Resultados de Acurácia do ResNet: {rn.tolist()}")
        def resumo(nome, x): print(f"{nome}: média={np.mean(x):.4f} | desvio={np.std(x, ddof=1) if len(x) > 1 else 0.0:.4f} | n={len(x)}")
        resumo("MyCNN", mc); resumo("ResNet50", rn)
        if len(mc) == len(rn) and len(mc) > 1:
            try:
                stat, p_val = wilcoxon(mc, rn, zero_method="zsplit")
                print(f"Wilcoxon -> estatística={stat:.4f} | p={p_val:.4f}")
                if p_val < 0.05: print("\nA diferença é estatisticamente significativa.")
                else: print("\nA diferença NÃO é estatisticamente significativa.")
            except Exception as e:
                print(f"Não foi possível rodar o teste de Wilcoxon: {e}")

    print("\n\n========== TREINAMENTO FINAL E AVALIAÇÃO NO CONJUNTO DE TESTE ==========")
    full_train_loader = get_data_loader(imgs_path=[data_path + path for path in train_csv_full['image_path']], labels=list(train_csv_full['pathology']), transform=get_val_transforms(), batch_size=batch_size, shuf=True)
    test_loader = get_data_loader(imgs_path=[data_path + path for path in test_csv['image_path']], labels=list(test_csv['pathology']), transform=get_val_transforms(), batch_size=batch_size, shuf=False)

    print("\n\n========== TREINAMENTO E AVALIAÇÃO FINAL DO MyCNN ==========")
    final_mycnn_model = MyCNN(num_classes=2).to(device)
    criterion_mycnn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_mycnn = optim.Adam(final_mycnn_model.parameters(), lr=learning_rate)
    trained_mycnn, _ = train(
        model=final_mycnn_model, train_loader=full_train_loader, criterion=criterion_mycnn, optimizer=optimizer_mycnn,
        device=device, num_epochs=num_epochs, grad_clip_norm=GRAD_CLIP_NORM, accum_steps=ACCUM_STEPS, use_amp=USE_AMP
    )
    test(trained_mycnn, test_loader, criterion_mycnn, device)
    os.makedirs("models", exist_ok=True)
    torch.save(trained_mycnn.state_dict(), "models/my_cnn_final.pth")
    print("\nMyCNN final treinado e salvo em models/my_cnn_final.pth")

    if RUN_RESNET:
        print("\n\n========== TREINAMENTO E AVALIAÇÃO FINAL DO ResNet50 ==========")
        final_resnet_model = get_resnet_model(num_classes=2, pretrained=True).to(device)
        criterion_resnet = nn.CrossEntropyLoss(weight=class_weights)
        optimizer_resnet = optim.Adam(final_resnet_model.parameters(), lr=learning_rate)
        trained_resnet, _ = train(
            model=final_resnet_model, train_loader=full_train_loader, criterion=criterion_resnet, optimizer=optimizer_resnet,
            device=device, num_epochs=num_epochs, grad_clip_norm=GRAD_CLIP_NORM, accum_steps=ACCUM_STEPS, use_amp=USE_AMP
        )
        test(trained_resnet, test_loader, criterion_resnet, device)
        torch.save(trained_resnet.state_dict(), "models/resnet50_final.pth")
        print("\nResNet50 final treinado e salvo em models/resnet50_final.pth")

if __name__ == "__main__":
    main()