import pandas as pd 

class DataIngestion:
    def __init__(self) -> None:
        self.filepath = None

    
    def load_data(self,filepath:str) -> pd.DataFrame:
        self.filepath = filepath

        df = pd.read_csv(self.filepath,encoding='latin-1')

        return df
