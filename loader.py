from torch.utils import data
import pandas as pd
import os

class MyDataset (data.Dataset):
    def __init__(self, jpeg_path, csv_path, dicom_info_filename, mass_train_filename, calc_train_filename, mass_test_filename, calc_test_filename):
        
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
        self.pacients = []

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

        print(result.columns)

        ### Preenche self.pacients
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
        self.pacients.append(imgs)

        print(self.pacients[0])




myDataset = MyDataset("./CBIS-DDSM/jpeg/", 
                      "./csv/", 
                      "dicom_info.csv",
                      "mass_case_description_train_set.csv", 
                      "calc_case_description_train_set.csv", 
                      "mass_case_description_test_set.csv",
                      "calc_case_description_test_set.csv")

