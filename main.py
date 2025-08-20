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
from training.validate import validate
from training.test import test
from utils.augmentation import get_train_transforms, get_val_transforms


def load_data(csv_path, dicom_info_filename,
              mass_train_filename, calc_train_filename,
              mass_test_filename, calc_test_filename,
              data_path="./data/"):
    """
    Carrega e prepara os CSVs, faz merges com informações DICOM e retorna
    os DataFrames de treino e teste com coluna 'image_path' relativa à pasta data/.
    """
    # Carrega os arquivos CSV
    dicom_info = pd.read_csv(os.path.join(csv_path, dicom_info_filename))
    mass_train = pd.read_csv(os.path.join(csv_path, mass_train_filename))
    calc_train = pd.read_csv(os.path.join(csv_path, calc_train_filename))
    mass_test = pd.read_csv(os.path.join(csv_path, mass_test_filename))
    calc_test = pd.read_csv(os.path.join(csv_path, calc_test_filename))

    # Renomeia colunas para padronização
    mass_train.rename(columns={"patient_id": "patient id", "breast_density": "breast density"}, inplace=True)
    calc_train.rename(columns={"patient_id": "patient id"}, inplace=True)
    mass_test.rename(columns={"patient_id": "patient id", "breast_density": "breast density"}, inplace=True)
    calc_test.rename(columns={"patient_id": "patient id"}, inplace=True)

    # Filtra informações DICOM para apenas imagens cortadas
    dicom_info = dicom_info[dicom_info["SeriesDescription"] == "cropped images"]

    # Concatena os dados de massas e calcificações
    train_data_csv = pd.concat([mass_train, calc_train], ignore_index=True)
    test_data_csv = pd.concat([mass_test, calc_test], ignore_index=True)

    # Extrai IDs únicos dos caminhos de arquivo para mesclagem
    train_data_csv["SeriesInstanceUID1"] = train_data_csv["image file path"].str.split('/').str[2]
    train_data_csv["SeriesInstanceUID2"] = train_data_csv["cropped image file path"].str.split('/').str[2]
    test_data_csv["SeriesInstanceUID1"] = test_data_csv["image file path"].str.split('/').str[2]
    test_data_csv["SeriesInstanceUID2"] = test_data_csv["cropped image file path"].str.split('/').str[2]

    # Mescla os dados de treino e teste com as informações DICOM
    merge1 = pd.merge(dicom_info, train_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID1", how='inner')
    merge2 = pd.merge(dicom_info, train_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID2", how='inner')
    merge3 = pd.merge(dicom_info, test_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID1", how='inner')
    merge4 = pd.merge(dicom_info, test_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID2", how='inner')

    # Combina os resultados e remove duplicatas
    train_csv_merged = pd.concat([merge1, merge2], ignore_index=True).drop_duplicates()
    test_csv_merged = pd.concat([merge3, merge4], ignore_index=True).drop_duplicates()

    # Limpa o caminho das imagens para garantir o carregamento correto
    train_csv_merged['image_path'] = train_csv_merged['image_path'].str.replace('CBIS-DDSM/', '', regex=False)
    test_csv_merged['image_path'] = test_csv_merged['image_path'].str.replace('CBIS-DDSM/', '', regex=False)

    return train_csv_merged, test_csv_merged


def main():
    # --- Configurações ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    num_epochs = 30
    learning_rate = 0.001
    batch_size = 32
    data_path = "./data/"
    csv_path = "./data/csv/"
    ISDEVELOPING = False

    # >>> Ajustes de gradiente <<<
    GRAD_CLIP_NORM = 1.0     # defina None para desativar clipping
    ACCUM_STEPS = 1          # >1 para acumular gradientes
    USE_AMP = None           # None=auto (ativa se CUDA), True/False para forçar

    # --- Configurações da Validação Cruzada ---
    N_SPLITS = 3
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

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
    
    # --- Cria um DataFrame de pacientes únicos com rótulos canônicos ---
    # Atribui o rótulo mais severo (maligno=1) a cada paciente
    patient_info = train_csv_full.groupby('patient id')['pathology'].max().reset_index()
    
    # --- Reduz o dataset para testes ---
    if (ISDEVELOPING):
        print ("\n----------------- ATENÇÃO: VOCE ESTÁ RODANDO COM APENAS PARTE DOS DADOS PARA DESENVOLVIMENTO -----------------\n")
        DEV_FRAC = 0.05
        # Amostragem do DataFrame de pacientes únicos
        temp_df_sampled = patient_info.sample(frac=DEV_FRAC, random_state=42)
        
        # Filtra os DataFrames completos com base nos pacientes selecionados
        train_csv_full = train_csv_full[train_csv_full['patient id'].isin(temp_df_sampled['patient id'])].reset_index(drop=True)
        test_csv = test_csv.sample(frac=DEV_FRAC, random_state=42).reset_index(drop=True)

        print("Tamanhos (após amostragem):",
            f"train={len(train_csv_full)} | test={len(test_csv)}")
        print("Distribuição de classes (train):")
        print(train_csv_full['pathology'].value_counts())
        print("Distribuição de classes (test):")
        print(test_csv['pathology'].value_counts())

        # Recria o patient_info a partir do DataFrame reduzido para que a estratificação seja correta
        patient_info = train_csv_full.groupby('patient id')['pathology'].max().reset_index()

    # --- Extrai IDs de paciente e rótulos ---
    patient_ids = patient_info['patient id']
    patient_labels = patient_info['pathology']

    # Calcule os pesos uma única vez com base no dataset de treino completo
    pathology_counts = train_csv_full['pathology'].value_counts()
    pathology_counts_dict = pathology_counts.to_dict()
    count_class_0 = pathology_counts_dict.get(0, 1)
    count_class_1 = pathology_counts_dict.get(1, 1)

    # O peso é o inverso da frequência da classe
    weight_class_0 = (count_class_0 + count_class_1) / count_class_0
    weight_class_1 = (count_class_0 + count_class_1) / count_class_1
    class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32).to(device)
    print(f"Pesos das classes calculados: {class_weights}")

    # --- Validação Cruzada do MyCNN ---
    print("\n\n========== VALIDANDO MyCNN COM VALIDAÇÃO CRUZADA ==========")
    mycnn_fold_results = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(patient_ids, y=patient_labels)):
        print(f"\n========== FOLD {fold + 1}/{N_SPLITS} ==========")
        
        # Obtem os IDs de paciente de treino e validação
        train_patient_ids = patient_ids.iloc[train_indices]
        val_patient_ids = patient_ids.iloc[val_indices]

        # Filtra o DataFrame original com base nos IDs
        train_df_fold = train_csv_full[train_csv_full['patient id'].isin(train_patient_ids)]
        val_df_fold = train_csv_full[train_csv_full['patient id'].isin(val_patient_ids)]

        # Verifique se o vazamento foi evitado
        common_patients = set(train_df_fold['patient id']).intersection(set(val_df_fold['patient id']))
        if common_patients:
            raise ValueError(f"Vazamento de dados detectado! Pacientes em comum: {common_patients}")

        train_loader = get_data_loader(
            imgs_path=[data_path + path for path in train_df_fold['image_path']],
            labels=list(train_df_fold['pathology']),
            transform=get_train_transforms(), batch_size=batch_size, shuf=True
        )
        val_loader = get_data_loader(
            imgs_path=[data_path + path for path in val_df_fold['image_path']],
            labels=list(val_df_fold['pathology']),
            transform=get_val_transforms(), batch_size=batch_size, shuf=False
        )

        # Treina um novo MyCNN em cada fold
        model = MyCNN(num_classes=2)
        model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        trained_model, _ = train(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            grad_clip_norm=GRAD_CLIP_NORM,
            accum_steps=ACCUM_STEPS,
            use_amp=USE_AMP
        )

        _, val_acc = validate(trained_model, val_loader, criterion, device)
        mycnn_fold_results.append(val_acc.item())
        print(f"Resultado do Fold {fold + 1}: Acurácia de Validação = {val_acc.item():.4f}")

    # --- Validação Cruzada do ResNet50 ---
    print("\n\n========== VALIDANDO ResNet50 COM VALIDAÇÃO CRUZADA ==========")
    resnet_fold_results = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(patient_ids, y=patient_labels)):
        print(f"\n========== FOLD {fold + 1}/{N_SPLITS} ==========")


        # Obtem os IDs de paciente de treino e validação
        train_patient_ids = patient_ids.iloc[train_indices]
        val_patient_ids = patient_ids.iloc[val_indices]

        # Filtra o DataFrame original com base nos IDs
        train_df_fold = train_csv_full[train_csv_full['patient id'].isin(train_patient_ids)]
        val_df_fold = train_csv_full[train_csv_full['patient id'].isin(val_patient_ids)]

        # Verifique se o vazamento foi evitado
        common_patients = set(train_df_fold['patient id']).intersection(set(val_df_fold['patient id']))
        if common_patients:
            raise ValueError(f"Vazamento de dados detectado! Pacientes em comum: {common_patients}")
                
        train_loader = get_data_loader(
            imgs_path=[data_path + path for path in train_df_fold['image_path']],
            labels=list(train_df_fold['pathology']),
            transform=get_train_transforms(), batch_size=batch_size, shuf=True
        )
        val_loader = get_data_loader(
            imgs_path=[data_path + path for path in val_df_fold['image_path']],
            labels=list(val_df_fold['pathology']),
            transform=get_val_transforms(), batch_size=batch_size, shuf=False
        )

        # Treina um novo ResNet em cada fold
        model = get_resnet_model(num_classes=2, pretrained=True)
        model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        trained_model, _ = train(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            grad_clip_norm=GRAD_CLIP_NORM,
            accum_steps=ACCUM_STEPS,
            use_amp=USE_AMP
        )

        _, val_acc = validate(trained_model, val_loader, criterion, device)
        resnet_fold_results.append(val_acc.item())
        print(f"Resultado do Fold {fold + 1}: Acurácia de Validação = {val_acc.item():.4f}")

    # --- Comparação Estatística ---
    print("\n\n========== TESTE ESTATÍSTICO ==========")
    mc = np.asarray(mycnn_fold_results, dtype=float)
    rn = np.asarray(resnet_fold_results, dtype=float)

    print(f"Resultados de Acurácia do MyCNN: {mc.tolist()}")
    print(f"Resultados de Acurácia do ResNet: {rn.tolist()}")

    # Resumo descritivo
    def resumo(nome, x):
        desvio = np.std(x, ddof=1) if len(x) > 1 else 0.0
        print(f"{nome}: média={np.mean(x):.4f} | desvio={desvio:.4f} | n={len(x)}")

    resumo("MyCNN", mc)
    resumo("ResNet50", rn)

    if len(mc) != len(rn) or len(mc) == 0:
        print("Conjuntos com tamanhos diferentes ou vazios — pulando teste estatístico.")
        statistic, p_value = None, None
    else:
        diffs = mc - rn
        if np.allclose(diffs, 0):
            print("Os resultados por fold são idênticos (todas as diferenças = 0).")
            print("Wilcoxon não aplicável; assumindo estatística=0 e p=1.0000.")
            statistic, p_value = 0.0, 1.0
        else:
            try:
                # 'zsplit' lida melhor com empates/zeros do que 'wilcox'/'pratt'
                statistic, p_value = wilcoxon(mc, rn, zero_method="zsplit")
                print(f"Wilcoxon -> estatística={statistic:.4f} | p={p_value:.4f}")
            except Exception as e:
                print(f"Wilcoxon falhou ({e}). Tentando t pareado...")
                stat_t, p_t = ttest_rel(mc, rn)
                statistic, p_value = stat_t, p_t
                print(f"T-test pareado -> estatística={statistic:.4f} | p={p_value:.4f}")

        if p_value is not None:
            if p_value < 0.05:
                print("\nA diferença de desempenho é estatisticamente significativa (p < 0.05).")
            else:
                print("\nA diferença de desempenho NÃO é estatisticamente significativa (p >= 0.05).")
        if len(mc) < 5:
            print("Aviso: n de folds muito pequeno; o poder estatístico é baixo.")

    # --- Treinamento e Avaliação Final dos Dois Modelos ---
    print("\n\n========== TREINAMENTO FINAL E AVALIAÇÃO NO CONJUNTO DE TESTE ==========")
    full_train_loader = get_data_loader(
        imgs_path=[data_path + path for path in train_csv_full['image_path']],
        labels=list(train_csv_full['pathology']),
        transform=get_train_transforms(), batch_size=batch_size, shuf=True
    )

    # DataLoader do conjunto de teste
    test_loader = get_data_loader(
        imgs_path=[data_path + path for path in test_csv['image_path']],
        labels=list(test_csv['pathology']),
        transform=get_val_transforms(), batch_size=batch_size, shuf=False
    )

    # === MODELO MyCNN ===
    print("\n\n========== TREINAMENTO E AVALIAÇÃO FINAL DO MyCNN ==========")
    final_mycnn_model = MyCNN(num_classes=2)
    final_mycnn_model.to(device)
    criterion_mycnn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_mycnn = optim.Adam(final_mycnn_model.parameters(), lr=learning_rate)

    trained_mycnn, _ = train(
        model=final_mycnn_model,
        dataloader=full_train_loader,
        criterion=criterion_mycnn,
        optimizer=optimizer_mycnn,
        device=device,
        num_epochs=num_epochs,
        grad_clip_norm=GRAD_CLIP_NORM,
        accum_steps=ACCUM_STEPS,
        use_amp=USE_AMP
    )

    test(trained_mycnn, test_loader, criterion_mycnn, device)
    os.makedirs("models", exist_ok=True)
    torch.save(trained_mycnn.state_dict(), "models/my_cnn_final.pth")
    print("\n✅ MyCNN final treinado e salvo em models/my_cnn_final.pth")

    # === MODELO ResNet50 ===
    print("\n\n========== TREINAMENTO E AVALIAÇÃO FINAL DO ResNet50 ==========")
    final_resnet_model = get_resnet_model(num_classes=2, pretrained=True)
    final_resnet_model.to(device)
    criterion_resnet = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_resnet = optim.Adam(final_resnet_model.parameters(), lr=learning_rate)

    trained_resnet, _ = train(
        model=final_resnet_model,
        dataloader=full_train_loader,
        criterion=criterion_resnet,
        optimizer=optimizer_resnet,
        device=device,
        num_epochs=num_epochs,
        grad_clip_norm=GRAD_CLIP_NORM,
        accum_steps=ACCUM_STEPS,
        use_amp=USE_AMP
    )

    test(trained_resnet, test_loader, criterion_resnet, device)
    torch.save(trained_resnet.state_dict(), "models/resnet50_final.pth")
    print("\n✅ ResNet50 final treinado e salvo em models/resnet50_final.pth")


if __name__ == "__main__":
    main()