from torch.utils import data
from PIL import Image
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class MyDataset (data.Dataset):
    def __init__(self, imgs_path, csv_path, dicom_info_filename, mass_train_filename, calc_train_filename, mass_test_filename, calc_test_filename, transform=None):
        
        super().__init__()
        '''
        patients: [[{
            'SeriesInstanceUID': "",
            'image_path': "",
            'pathology': "",
            'Laterality': "",
            'SeriesDescription': "",
            'PatientOrientation': "",
            'abnormality type': ""
        }]]
        '''
        self.patients = []

        ### Load CSV
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

        ### Remove ROI mask
        dicom_info = dicom_info[dicom_info["SeriesDescription"] != "ROI mask images"]

        ### Concat dataframes
        all_data = pd.concat([mass_train, calc_train, mass_test, calc_test], ignore_index=True)

        ### Merge dicom_info e all_data
        # Cria colunas para Merge
        dicom_info["SeriesInstanceUID2"] = dicom_info["SeriesInstanceUID"]
        dicom_info["SeriesInstanceUID1"] = dicom_info["SeriesInstanceUID"]
        all_data["SeriesInstanceUID1"] = all_data["image file path"].str.split('/').str[2]
        all_data["SeriesInstanceUID2"] = all_data["cropped image file path"].str.split('/').str[2]

        # Merge usando SeriesInstanceUID1
        merge1 = pd.merge(dicom_info, all_data, left_on="SeriesInstanceUID1", right_on="SeriesInstanceUID1", how='inner')

        # Merge usando SeriesInstanceUID2
        merge2 = pd.merge(dicom_info, all_data, left_on="SeriesInstanceUID2", right_on="SeriesInstanceUID2", how='inner')

        # Concatenar os dois resultados e remover duplicatas
        result = pd.concat([merge1, merge2], ignore_index=True).drop_duplicates()

        ### Preenche self.patients
        # Agrupa pelo paciente
        for _, group in result.groupby('patient id'):
            imgs = []
            for _, row in group.iterrows():
                img_data = {
                    'SeriesInstanceUID': row.get('SeriesInstanceUID', None),
                    'image_path': row.get('image_path', None),
                    'pathology': row.get('pathology', None),
                    'Laterality': row.get('Laterality', None),
                    'SeriesDescription': row.get('SeriesDescription', None),
                    'PatientOrientation': row.get('PatientOrientation', None),
                    'abnormality type': row.get('abnormality type', None)
                }
                imgs.append(img_data)
        self.patients.append(imgs)

        self.imgs_path = imgs_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.patients)


    def get_image_by_patient(self, patient_index, image_index):
        """
        It gets the image, labels and meta-data (if applicable) according to the index informed in `image_index`.
        It also performs the transform on the image.

        :param image_index (int): an index in the interval [0, ..., len(img_paths)-1]
        :return (tuple): a tuple containing the image, its label and meta-data (if applicable)
        """

        image = Image.open(os.path.join (self.imgs_path, self.patients[patient_index][image_index]['image_path'])).convert("RGB")
            
        if self.transform is not None:
            image = self.transform(image)

        img_id = self.patients[patient_index][image_index]['SeriesInstanceUID']      
        labels = self.patients[patient_index][image_index]['pathology']
        meta_data = self.patients[patient_index][image_index]

        return image, labels, meta_data, img_id



myDataset = MyDataset("./", 
                      "./csv/", 
                      "dicom_info.csv",
                      "mass_case_description_train_set.csv", 
                      "calc_case_description_train_set.csv", 
                      "mass_case_description_test_set.csv",
                      "calc_case_description_test_set.csv")

image, label, meta_data, uid = myDataset.get_image_by_patient(0, 0)

if isinstance(image, torch.Tensor):
    # Convert tensor to PIL image for saving
    from torchvision.transforms.functional import to_pil_image
    image = to_pil_image(image)

print()
print("#"*30, "LOADER", "#"*30)
print("TESTE: Abrindo info de paciente 0 e imagem 0")
image.save("output_image.jpg")
print("Label:", label)
print("UID:", uid)
print("Meta-data:", meta_data)
print("Image saved as output_image.jpg")
print("#"*68)
print()
