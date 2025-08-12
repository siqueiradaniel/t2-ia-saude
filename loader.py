from torch.utils import data
import pandas as pd
import os

class MyDataset (data.Dataset):
    def __init__(self, jpeg_path, csv_path, dicom_info_filename, mass_train_filename, calc_train_filename, mass_test_filename, calc_test_filename):
        
        super().__init__()
        # patients: { pacient_id: ( SeriesInstaceUID, image_path, pathology, Laterality, SeriesDescription, PatientOrientation, abnormality (‘calc’ or ‘mass’) )}
        self.pacients = dict()

        # Load CSV
        mass_train = pd.read_csv (os.path.join (csv_path, mass_train_filename))
        calc_train = pd.read_csv (os.path.join (csv_path, calc_train_filename))
        mass_test = pd.read_csv (os.path.join (csv_path, mass_test_filename))
        calc_test = pd.read_csv (os.path.join (csv_path, calc_test_filename))

        # Rename columns to a pattern
        mass_train.rename (columns= {"patient_id" : "patient id", "breast_density" : "breast density"}, inplace=True)
        calc_train.rename (columns= {"patient_id" : "patient id"}, inplace=True)
        mass_test.rename (columns= {"patient_id" : "patient id", "breast_density" : "breast density"}, inplace=True)
        calc_test.rename (columns= {"patient_id" : "patient id"}, inplace=True)

        # Concat dataframes
        all_data = pd.concat([mass_train, calc_train, mass_test, calc_test], ignore_index=True)

        # Join entre dicom_info e all_data para unir informações necessárias
        dicom_info = pd.read_csv(os.path.join (csv_path, dicom_info_filename))

        print(dicom_info.columns, all_data.columns)
        # print(mass_train.columns, calc_train.columns, mass_test.columns, calc_test.columns)
        # print(all_data.columns)
        # print(all_data.shape)

        # Merge

        



        # preenche pacients

    





myDataset = MyDataset("./CBIS-DDSM/jpeg/", 
                      "./csv/", 
                      "dicom_info.csv",
                      "mass_case_description_train_set.csv", 
                      "calc_case_description_train_set.csv", 
                      "mass_case_description_test_set.csv",
                      "calc_case_description_test_set.csv")

