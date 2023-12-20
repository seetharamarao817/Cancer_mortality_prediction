from data_ingestion import DataIngestion
from handling_outliers import find_outliers
from model_building import (split_dataset,lr_model,metrices)
from scipy.stats import normaltest

di = DataIngestion()
df = di.load_data("/workspaces/Cancer_prediction/data/cancer_preprocessed.csv")

_,capped_data = find_outliers(df)

X_train, X_test, y_train, y_test = split_dataset(capped_data)

lrm = lr_model(X_train,y_train)

print(metrices(lrm,X_test,y_test))