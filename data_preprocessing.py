import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing  
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.layers import Dropout
from sklearn.decomposition import PCA




pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def gendata(doPCA):

    # Create dataframe from csv
    data = pd.read_csv('https://raw.githubusercontent.com/microsoft/r-server-hospital-length-of-stay/master/Data/LengthOfStay.csv')
    #Save the length of stay in a different variable
    labels = data['lengthofstay']
    # Drop columns that we dont need like specific dates, or the id of the patient
    data = data.drop(["eid", "vdate", "discharged", "lengthofstay"], axis=1)
    # Add dummy encoding for the object and type variables
    # For example, turn gender column into 2 columns, where a male will be 1 in the first column
    # and a 0 in the second column, and a female will be the inverse
    data = pd.get_dummies(data, columns=['rcount'])
    data = pd.get_dummies(data, columns=['gender'])
    data = pd.get_dummies(data, columns=['facid'])

    hematocrit = data[['hematocrit']].values
    data['hematocrit'] = preprocessing.StandardScaler().fit_transform(hematocrit)

    bloodureanitro = data[['neutrophils']].values
    data['neutrophils'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

    sodium = data[['sodium']].values
    data['sodium'] = preprocessing.StandardScaler().fit_transform(sodium)

    glucose = data[['glucose']].values
    data['glucose'] = preprocessing.StandardScaler().fit_transform(glucose)

    bloodureanitro = data[['bloodureanitro']].values
    data['bloodureanitro'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

    creatinine = data[['creatinine']].values
    data['creatinine'] = preprocessing.StandardScaler().fit_transform(creatinine)

    bmi = data[['bmi']].values
    data['bmi'] = preprocessing.StandardScaler().fit_transform(bmi)

    pulse = data[['pulse']].values
    data['pulse'] = preprocessing.StandardScaler().fit_transform(pulse)

    respiration = data[['respiration']].values
    data['respiration'] = preprocessing.StandardScaler().fit_transform(respiration)


    # ADD PCA CONSIDERATIONS
    if doPCA:
        pca = PCA()   
        data=pca.fit_transform(data)
        train_X = np.array(data[:80000])
        train_Y = labels.head(n=80000).to_numpy()
        test_X = np.array(data[80000:])
        test_Y = labels.tail(n=20000).to_numpy()
        return train_X, test_X, train_Y, test_Y
        

    # Seperate for train and test
    train_X = data.head(n=80000).to_numpy()
    test_X = data.tail(n=20000).to_numpy()
    train_Y = labels.head(n=80000).to_numpy()
    test_Y = labels.tail(n=20000).to_numpy()


    return train_X, test_X, train_Y, test_Y




