import os
import pandas as pd


class Reader:

    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        patients = pd.DataFrame(pd.read_excel(self.file_path, "patients")).fillna(0).astype('int')
        nurses = pd.DataFrame(pd.read_excel(self.file_path, "nurses")).fillna(0).astype('int')
        return patients, nurses
    
    

if __name__ == "__main__":
    reader = Reader("./DB/100P.xlsx")
    patients, nurses = reader.read()
    print(patients.head())
    print(nurses.head())
