import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# Definindo a classe PatientDataset para sua estrutura
class PatientDataset(Dataset):
    def __init__(self, csv_path, base_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_path = base_path
        self.transform = transform
        
        # Agrupar por 'PatientID' para ter uma lista de imagens por paciente
        self.df['CleanPatientID'] = self.df['PatientID'].str.extract(r'(P_\d+)')
        self.grouped = self.df.groupby('CleanPatientID.')
        self.patients = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        
        # Obter os dados (linhas do CSV) para este paciente
        patient_data = self.grouped.get_group(patient_id)
        
        # Pegar todos os caminhos das imagens para este paciente
        image_paths = patient_data['image_path'].tolist()
        
        images = []
        for path in image_paths:
            # Construir o caminho completo da imagem. 
            # O base_path já inclui /CBIS-DDSM/jpeg, então o caminho final é `base_path` + `path`
            # Note que a coluna 'image_path' no seu exemplo já tem a estrutura 'CBIS-DDSM/jpeg/...'
            # A partir de 'dicom_info.csv' na raiz, o caminho para as imagens é 'CBIS-DDSM/jpeg/...'
            # Se você já está na raiz, o caminho é exatamente o que está na coluna.
            full_path = os.path.join(os.path.dirname(self.base_path), path)
            
            # Ajuste aqui para o seu caso específico.
            # Se 'dicom_info.csv' está na raiz e as imagens em /CBIS-DDSM/,
            # e a coluna 'image_path' contém 'CBIS-DDSM/jpeg/...', o caminho completo é o valor da coluna.
            # O exemplo abaixo assume que `self.base_path` é a raiz do seu projeto.
            
            try:
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except FileNotFoundError:
                print(f"Arquivo não encontrado: {full_path}")

        # Se não houver imagens, retorne tensores vazios
        if not images:
            return torch.Tensor(), patient_id

        # Agrupar as imagens em um tensor
        images_tensor = torch.stack(images)
        return images_tensor, patient_id

# ---
### **Exemplo de uso com sua estrutura**

if __name__ == '__main__':
    # A sua estrutura de arquivos é assim:
    # /dicom_info.csv
    # /CBIS-DDSM/jpeg/...

    # O caminho para o CSV na sua estrutura.
    csv_path = './csv/dicom_info.csv' 
    # A base_path deve ser o diretório principal do seu projeto, onde o CSV está.
    base_path = './' 

    # As transformações
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
    first_patient_images, first_patient_id = patient_dataset[0]
    print(f"\nID do primeiro paciente: {first_patient_id}")
    print(f"Número de imagens para este paciente: {len(first_patient_images)}")
    if len(first_patient_images) > 0:
        print(f"Formato da primeira imagem: {first_patient_images[0].shape}")

    # Acessando o segundo paciente
    second_patient_images, second_patient_id = patient_dataset[1]
    print(f"\nID do segundo paciente: {second_patient_id}")
    print(f"Número de imagens para este paciente: {len(second_patient_images)}")
    if len(second_patient_images) > 0:
        print(f"Formato da primeira imagem: {second_patient_images[0].shape}")