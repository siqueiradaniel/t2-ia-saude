from torch.utils import data
import pandas as pd
import os

class MyDataset (data.Dataset):
    def __init__(self, jpeg_path, csv_path, mass_train_file_name, calc_train_file_name, mass_test_file_name, calc_test_file_name):
        
        super().__init__()
        # patients: { pacient_id: ( SeriesInstaceUID, image_path, pathology, Laterality, SeriesDescription, PatientOrientation, abnormality (‘calc’ or ‘mass’) )}
        self.pacients = dict()


        jpeg = jpeg_path
        csv = csv_path

        mass_train = pd.read_csv (os.path.join (csv, mass_train_file_name))
        calc_train = pd.read_csv (os.path.join (csv, calc_train_file_name))
        mass_test = pd.read_csv (os.path.join (csv, mass_test_file_name))
        calc_test = pd.read_csv (os.path.join (csv, calc_test_file_name))

        print(mass_train)
        # Concatena 
        # Merge
        # preenche pacients



myDataset = MyDataset("./CBIS-DDSM/jpeg/", 
                      "./csv/", 
                      "mass_case_description_train_set.csv", 
                      "calc_case_description_train_set.csv", 
                      "mass_case_description_test_set.csv",
                      "calc_case_description_test_set.csv")

