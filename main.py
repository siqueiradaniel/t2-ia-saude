import os
import pandas as pd
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from dataset import loader as ld

##### EXEMPLO DE USO DO DATALOADER
# Load CSV
save_dir = "./data/result_imgs"
data_path = './data/'
csv_path="./data/csv/"
dicom_info_filename="dicom_info.csv"
mass_train_filename="mass_case_description_train_set.csv"
calc_train_filename="calc_case_description_train_set.csv"
mass_test_filename="mass_case_description_test_set.csv"
calc_test_filename="calc_case_description_test_set.csv"

dicom_info = pd.read_csv(os.path.join (csv_path, dicom_info_filename))
mass_train = pd.read_csv (os.path.join (csv_path, mass_train_filename))
calc_train = pd.read_csv (os.path.join (csv_path, calc_train_filename))
mass_test = pd.read_csv (os.path.join (csv_path, mass_test_filename))
calc_test = pd.read_csv (os.path.join (csv_path, calc_test_filename))

# Rename columns to a pattern
mass_train.rename (columns= {"patient_id" : "patient id", "breast_density" : "breast density"}, inplace=True)
calc_train.rename (columns= {"patient_id" : "patient id"}, inplace=True)
mass_test.rename (columns= {"patient_id" : "patient id", "breast_density" : "breast density"}, inplace=True)
calc_test.rename (columns= {"patient_id" : "patient id"}, inplace=True)

# Remove full mammogram images
dicom_info = dicom_info[dicom_info["SeriesDescription"] == "cropped images"]

# Concat dataframes
train_data_csv = pd.concat([mass_train, calc_train], ignore_index=True)
test_data_csv = pd.concat([mass_test, calc_test], ignore_index=True)

### Merge dicom_info e dados de treino ou teste
# Cria colunas para Merge
train_data_csv["SeriesInstanceUID1"] = train_data_csv["image file path"].str.split('/').str[2]
train_data_csv["SeriesInstanceUID2"] = train_data_csv["cropped image file path"].str.split('/').str[2]

# Merge usando SeriesInstanceUID
merge1 = pd.merge(dicom_info, train_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID1", how='inner')
merge2 = pd.merge(dicom_info, train_data_csv, left_on="SeriesInstanceUID", right_on="SeriesInstanceUID2", how='inner')

# Concatenar os dois resultados e remover duplicatas
train_csv_merged = pd.concat([merge1, merge2], ignore_index=True).drop_duplicates()

### DataLoader
# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Parametros
imgs_path = [data_path + path for path in train_csv_merged['image_path']]
labels = list(train_csv_merged['pathology'])
meta_data = train_csv_merged[
    ['Laterality', 'SeriesDescription', 'SeriesInstanceUID', 
     'patient id', 'breast density', 'abnormality type']
].to_dict(orient='records')


# Create DataLoader
loader = ld.get_data_loader (imgs_path, labels, meta_data, transform)

### Cria diretório imgs
os.makedirs(save_dir, exist_ok=True)

# Pegando um batch e salva localmente
# RETIRAR SALVAMENTO -------------------------------------------------------------------------
for images, labels, meta_data, uids in loader:
    print("Batch shape:", images.shape)
    print("Labels:", labels)
    print("UIDs:", uids)
    print("Metadata:\n", meta_data)

    for i in range(images.shape[0]):  # itera sobre o batch
        img = to_pil_image(images[i])  # converte tensor para PIL Image
        filename = f"{uids[i]}_{labels[i]}.jpg"  # nome do arquivo
        img.save(os.path.join(save_dir, filename))
    break
