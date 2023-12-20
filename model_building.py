from data_ingestion import DataIngestion
from sklearn.model_selection import train_test_split
from handling_outliers import find_outliers
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import statsmodels.api as sm


def split_dataset(data):
    dependent_fer = data['TARGET_deathRate']
    indep_fer = data.drop('TARGET_deathRate',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(dependent_fer,indep_fer, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def lr_model(x_train,y_train):
    x_train_with_intercept = sm.add_constant(x_train)
    lr = sm.OLS(y_train,x_train_with_intercept).fit()
    return lr

def metrices(model,x_test,y_test):
    x_test_with_intercept = sm.add_constant(x_test)
    y_pred = model.predict(x_test_with_intercept)

    print("The R2 score of the model is ",r2_score(y_test, y_pred))
    print("The Mean absolute error is ",mean_absolute_error(y_test,y_pred))
    print("The Mean squared error is ",mean_squared_error(y_test,y_pred))

