from data_ingestion import DataIngestion
from feature_engi import bin_to_num
from feature_engi import cal_col
from feature_engi import one_hot_enc
from data_cleaning import (del_dupli_rows,drop_nd_fill_na)



di = DataIngestion()
df = di.load_data("/workspaces/Cancer_prediction/data/cancer_reg.csv")

df = del_dupli_rows(df)
df = bin_to_num(df)
df = cal_col(df)
df = one_hot_enc(df)
df = drop_nd_fill_na(df)



df.to_csv("/workspaces/Cancer_prediction/data/cancer_preprocessed.csv",index = False)