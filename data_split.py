from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(data):
    dependent_fer = data['TARGET_deathRate']
    indep_fer = data.drop('TARGET_deathRate',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(indep_fer,dependent_fer,test_size=0.25, random_state=42)
    
    print("done reshaping")
    return X_train, X_test, y_train, y_test