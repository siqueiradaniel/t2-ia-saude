import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

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
        
        # 3. Criar uma chave de união mais robusta
        # A coluna 'SeriesInstanceUID' está presente em ambos os DataFrames
        # e identifica um conjunto de imagens (ex: a mamografia de uma mama).
        # Para os arquivos de descrição, a SeriesInstanceUID é o penúltimo diretório no path.
        all_desc_df['SeriesInstanceUID'] = all_desc_df['image file path'].str.split('/').str[-2]
        
        # 4. Mesclar com o DataFrame principal para adicionar 'pathology'
        self.df = pd.merge(self.df, all_desc_df[['SeriesInstanceUID', 'pathology']], 
                           on='SeriesInstanceUID', how='left', suffixes=('', '_desc'))
        
        # O merge pode criar colunas duplicadas.
        if 'pathology_desc' in self.df.columns:
            self.df['pathology'].fillna(self.df['pathology_desc'], inplace=True)
            self.df.drop(columns='pathology_desc', inplace=True)
            
        # 5. Aplicar o mapeamento de rótulos binários
        class_mapper = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}
        self.df["label"] = self.df["pathology"].str.upper().map(class_mapper)
        
        # 6. Agrupar por 'PatientID' limpo
        self.df['CleanPatientID'] = self.df['PatientID'].str.extract(r'(P_\d+)')
        self.grouped = self.df.groupby('CleanPatientID')
        self.patients = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        
        patient_data = self.grouped.get_group(patient_id)
        
        # Extrair os caminhos e os UIDs das imagens para este paciente
        image_paths = patient_data['image_path'].tolist()
        sop_instance_uids = patient_data['SOPInstanceUID'].tolist()
        
        images = []
        labels_dict = {}
        for path, sop_uid in zip(image_paths, sop_instance_uids):
            full_path = os.path.join(self.base_path, path)
            try:
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
                
                # Adicionar ao dicionário de rótulos
                # O rótulo para esta imagem é o da linha correspondente no DataFrame
                label = patient_data[patient_data['SOPInstanceUID'] == sop_uid]['label'].iloc[0]
                labels_dict[sop_uid] = label
            except FileNotFoundError:
                print(f"Arquivo não encontrado: {full_path}")
        
        if not images:
            return torch.Tensor(), patient_id, {}

        images_tensor = torch.stack(images)
        return images_tensor, patient_id, labels_dict

# ---
### **Exemplo de uso**

if __name__ == '__main__':
    csv_path = './csv/dicom_info.csv'
    base_path = './'
    transform_op = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    patient_dataset = PatientDataset(
        csv_path=csv_path,
        base_path=base_path,
        transform=transform_op
    )

    print(f"Total de pacientes no dataset: {len(patient_dataset)}")

    for i in range(2):
        try:
            images, p_id, labels_dict = patient_dataset[i]
            print(f"\nID do paciente: {p_id}")
            print(f"Número de imagens para este paciente: {len(images)}")
            if len(images) > 0:
                first_image_uid = list(labels_dict.keys())[0]
                print(f"Rótulos das imagens para este paciente:")
                for uid, label in labels_dict.items():
                    print(f"  - Imagem {uid}: {label} (0=Benigno, 1=Maligno)")
                print(f"Formato da primeira imagem: {images[0].shape}")
        except Exception as e:
            print(f"Erro ao processar o paciente {i}: {e}")