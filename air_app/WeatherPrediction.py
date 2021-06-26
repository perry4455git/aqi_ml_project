#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:59:05 2021


@author: PRAJWAL
"""


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
import os
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

def predict_weather(list):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('dir_path --- ', dir_path)
    data = pd.read_csv(dir_path+"/city_day.csv")

    data = data.dropna(subset=['AQI_Bucket'])

    data_a = data[['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','AQI','AQI_Bucket']]

    print(data_a.isna().sum())

    #handing missing values
    data_a['PM2.5'] = data_a['PM2.5'].fillna(data_a['PM2.5'].median())
    data_a['PM10'] = data_a['PM10'].fillna(data_a['PM10'].median())
    data_a['NO'] = data_a['NO'].fillna(data_a['NO'].median())
    data_a['NO2'] = data_a['NO2'].fillna(data_a['NO2'].median())
    data_a['NOx'] = data_a['NOx'].fillna(data_a['NOx'].median())
    data_a['NH3'] = data_a['NH3'].fillna(data_a['NH3'].median())
    data_a['CO'] = data_a['CO'].fillna(data_a['CO'].median())
    data_a['SO2'] = data_a['SO2'].fillna(data_a['SO2'].median())
    data_a['O3'] = data_a['O3'].fillna(data_a['O3'].median())
    data_a['Benzene'] = data_a['Benzene'].fillna(data_a['Benzene'].median())
    data_a['Toluene'] = data_a['Toluene'].fillna(data_a['Toluene'].median())

    data_a['AQI_Bucket'].value_counts()

    data_a['AQI_Bucket'] = data_a['AQI_Bucket'].replace(['Moderate', 'Satisfactory'], 'Good')
    data_a['AQI_Bucket'] = data_a['AQI_Bucket'].replace(['Very Poor', 'Severe'], 'Poor')

    data_a['AQI_Bucket'] = data_a['AQI_Bucket'].replace(['Good'],1)
    data_a['AQI_Bucket'] = data_a['AQI_Bucket'].replace(['Poor'],0)

    X = data_a.iloc[:,:-1]
    y = data_a.iloc[:,-1]
    print(X)
    print(y)
    #train_test_splitting of data
    #from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=100)

    #SVM training
    model1 = svm.SVC()
    model1.fit(X_train,y_train)
    svm_score=model1.score(X_test,y_test)
    print(svm_score)
    svm_predict = model1.predict(np.array(list).reshape(1,-1))
    print('svm_predict ---', svm_predict)
    if svm_predict== [1]:
        svm_p="1"
    else :
        svm_p="0"

    #KNN training
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    neigh_score=neigh.score(X_test,y_test)
    print(neigh_score)
    neigh_predict = neigh.predict(np.array(list).reshape(1,-1))
    print('neigh_predict ---', neigh_predict)
    if neigh_predict== [1]:
        neigh_p="1"
    else:
        neigh_p="0"
    #ANN training
    model2 = Sequential()
    model2.add(Dense(12, input_dim=12, activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(8, activation='relu'))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model2.fit(X, y, epochs=1, batch_size=10)

    print('X_test.iloc[10] ', np.array(X_test.iloc[10]).reshape(1,-1))
    #print('X_test.iloc[10] X_test ', X_test)
    #ann_predict = model2.predict(X_test)
    
    print('list ---', np.array(list).reshape(1,-1))
    ann_predict_accuracy = model2.predict(np.array(list).reshape(1,-1))
    print('ann_predict_accuracy ---', ann_predict_accuracy[0][0])
    print("new==",svm_p)
    print("nei==",neigh_p)

    return ann_predict_accuracy[0][0]*100, svm_score*100, neigh_score*100,svm_p,neigh_p

'''
def output_results(list):
    list_of_outputs = []
    list_of_outputs.append(model1.predict(np.array(list).reshape(1,-1))
    list_of_outputs.append(neigh.predict(np.array(list).reshape(1,-1))
    list_of_outputs.append(model2.predict(np.array(list).reshape(1,-1))
                           print(list_of_outputs)
    return list_of_outputs

'''


def predict_aqi(list):
    # loading dataset and storing in train variable
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('dir_path --- ', dir_path)
    data = pd.read_csv(dir_path + "/city_day.csv")

    data = data.dropna(subset=['AQI_Bucket'])

    data_a = data[['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3', 'AQI']]

    print(data_a.isna().sum())

    # handing missing values
    data_a['PM2.5'] = data_a['PM2.5'].fillna(data_a['PM2.5'].median())
    data_a['PM10'] = data_a['PM10'].fillna(data_a['PM10'].median())
    data_a['NO2'] = data_a['NO2'].fillna(data_a['NO2'].median())
    data_a['NH3'] = data_a['NH3'].fillna(data_a['NH3'].median())
    data_a['SO2'] = data_a['SO2'].fillna(data_a['SO2'].median())
    data_a['CO'] = data_a['CO'].fillna(data_a['CO'].median())
    data_a['O3'] = data_a['O3'].fillna(data_a['O3'].median())
    data_a['AQI'] = data_a['AQI'].fillna(data_a['AQI'].median())

    train = data_a

    # display top 5 data
    train.head()

    # creating model
    m1 = RandomForestRegressor()

    # separating class label and other attributes
    train1 = train.drop(['AQI'], axis=1)
    target = train['AQI']

    # Fitting the model
    m1.fit(train1, target)
    '''RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        max_samples=None, min_impurity_decrease=0.0,
                        min_impurity_split=None, min_samples_leaf=1,
                        min_samples_split=2, min_weight_fraction_leaf=0.0,
                        n_estimators=100, n_jobs=None, oob_score=False,
                        random_state=None, verbose=0, warm_start=False)'''

    # calculating the score and the score is  97.96360799890066%
    m1.score(train1, target) * 100

    # predicting the model with other values (testing the data)
    # so AQI is 123.71
    m1.predict([[123, 45, 67, 34, 5, 0, 23]])
    random_forest_aqi = m1.predict(np.array(list).reshape(1, -1))
    print('random_forest_aqi --- ', random_forest_aqi)

    '''
    # Adaboost model
    # importing module

    # defining model
    m2 = AdaBoostRegressor()

    # Fitting the model
    m2.fit(train1, target)

    #AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear', n_estimators=50, random_state=None)

    # calculating the score and the score is  96.15377360010211%
    m2.score(train1, target)*100

    # predicting the model with other values (testing the data)
    # so AQI is 94.42105263
    ada_aqi = m2.predict(np.array(list).reshape(1,-1))
    print('ada_aqi --- ', ada_aqi)
    '''

    return random_forest_aqi

'''
if __name__ == "__main__":
    list = [123, 45, 67, 34, 5, 0, 23]

    predict_aqi(list)
'''