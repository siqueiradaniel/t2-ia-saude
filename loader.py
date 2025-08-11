import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# Definindo a classe PatientDataset
class PatientDataset(Dataset):
    def __init__(self, csv_path, base_path, transform=None):
        # 1. Carregar o CSV principal
        self.df = pd.read_csv(csv_path)
        self.base_path = base_path
        self.transform = transform
        
        # 2. Carregar e concatenar os arquivos de descrição
        desc_files = [
            './csv/mass_case_description_train_set.csv',
            './csv/mass_case_description_test_set.csv',
            './csv/calc_case_description_train_set.csv',
            './csv/calc_case_description_test_set.csv'
        ]
        
        all_desc_df = pd.DataFrame()
        for f in desc_files:
            if os.path.exists(f):
                desc_df = pd.read_csv(f)
                all_desc_df = pd.concat([all_desc_df, desc_df], ignore_index=True)
        
        # 3. Normalizar e criar a chave de mesclagem para o dataframe de descrição
        # Extrair a chave 'Mass-Training_P_00001_LEFT_CC' do path
        all_desc_df['merge_key'] = all_desc_df['image file path'].str.split('/').str[0].str.strip()

        print(all_desc_df.shape)
        
        # 4. Mesclar com o DataFrame principal para adicionar 'pathology'
        # A chave de mesclagem no DataFrame principal já é a coluna 'PatientID'
        self.df = pd.merge(self.df, all_desc_df[['merge_key', 'pathology']], 
                           left_on='PatientID', right_on='merge_key', how='left')
        self.df.drop(columns=['merge_key'], inplace=True)
        
        # 5. Aplicar o mapeamento de rótulos binários
        class_mapper = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}
        self.df["label"] = self.df["pathology"].str.upper().map(class_mapper)
        
        # 6. Agrupar por PatientID limpo
        self.df['CleanPatientID'] = self.df['PatientID'].str.extract(r'(P_\d+)')
        self.grouped = self.df.groupby('CleanPatientID')
        self.patients = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        
        patient_data = self.grouped.get_group(patient_id)
        image_paths = patient_data['image_path'].tolist()
        
        images = []
        for path in image_paths:
            full_path = os.path.join(self.base_path, path)
            try:
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except FileNotFoundError:
                print(f"Arquivo não encontrado: {full_path}")

        if not images:
            return torch.Tensor(), patient_id

        images_tensor = torch.stack(images)
        patient_label = patient_data['label'].iloc[0]
        return images_tensor, patient_id, patient_label

# ---
### **Exemplo de uso**

if __name__ == '__main__':
    # Seus caminhos aqui (ajuste conforme necessário)
    csv_path = './csv/dicom_info.csv' 
    base_path = './' 

    transform_op = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Instanciando o dataset
    patient_dataset = PatientDataset(
        csv_path=csv_path,
        base_path=base_path,
        transform=transform_op
    )

    print(f"Total de pacientes no dataset: {len(patient_dataset)}")

    # Acessando o primeiro paciente
    
    for i in range(5):
        first_patient_images, first_patient_id, first_patient_label = patient_dataset[i]
        print(f"\nID do primeiro paciente: {first_patient_id}")
        print(f"Rótulo do primeiro paciente (0=Benigno, 1=Maligno): {first_patient_label}")
        print(f"Número de imagens para este paciente: {len(first_patient_images)}")
        if len(first_patient_images) > 0:
            print(f"Formato da primeira imagem: {first_patient_images[i].shape}")




