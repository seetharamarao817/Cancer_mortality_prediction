from data_ingestion import DataIngestion
from handling_outliers import find_outliers
from data_split import split_dataset
from scipy.stats import normaltest
from model_building import train_model
from model_evaluation import evaluation

di = DataIngestion()
df = di.load_data("/workspaces/Cancer_mortality_prediction/data/cancer_preprocessed.csv")
print("done ingestion")
_,capped_data = find_outliers(df)
print("done handling outliers")
X_train, X_test, y_train, y_test = split_dataset(capped_data)
print("done spliting")
model = train_model(X_train, X_test, y_train, y_test,model_name="randomforest")
print("done training")
r2_score, rmse = evaluation(model,X_test,y_test)
print(r2_score)
print(rmse)
