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
        
        # 3. Criar o dicionário de busca para patologia
        # A chave para o dicionário é o nome do diretório do paciente,
        # que corresponde exatamente ao 'PatientID' no dicom_info.csv
        pathology_dict = {}
        for index, row in all_desc_df.iterrows():
            # Tentar usar o 'image file path' primeiro
            if pd.notna(row['image file path']):
                key = row['image file path'].split('/')[0]
                pathology_dict[key] = row['pathology']
            
            # Se não houver, usar o 'cropped image file path'
            if pd.notna(row['cropped image file path']):
                key = row['cropped image file path'].split('/')[0]
                pathology_dict[key] = row['pathology']
        
        # 4. Usar o dicionário para adicionar a coluna 'pathology' ao DataFrame principal
        # A coluna 'PatientID' no seu df já é a chave que precisamos
        self.df['pathology'] = self.df['PatientID'].map(pathology_dict)
        
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
            # Retornar o rótulo nan se não houver imagens
            return torch.Tensor(), patient_id, float('nan')

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

    patient_dataset = PatientDataset(
        csv_path=csv_path,
        base_path=base_path,
        transform=transform_op
    )

    print(f"Total de pacientes no dataset: {len(patient_dataset)}")

    for i in range(5):
        try:
            images, p_id, label = patient_dataset[i]
            print(f"\nID do paciente: {p_id}")
            print(f"Rótulo (0=Benigno, 1=Maligno): {label}")
            print(f"Número de imagens: {len(images)}")
            if len(images) > 0:
                print(f"Formato da primeira imagem: {images[0].shape}")
        except Exception as e:
            print(f"Erro ao processar o paciente {i}: {e}")