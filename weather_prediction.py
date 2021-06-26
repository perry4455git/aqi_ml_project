import pandas as pd
import numpy as np
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
print('dir_path --- ', dir_path)
csv_path = dir_path+"/city_day.csv"

#Import the dataset (We are using 'datasets/city_day.csv' )

air_pollution_type_dataset = pd.read_csv(csv_path, sep = ",")



#Initiate missing values formats

missing_value_formats = ["nan","n.a.","?","NA","n/a", "na", "--"]





#Import the dataset

airquality_df = pd.read_csv(csv_path, sep = ",", na_values = missing_value_formats)



# Null Values in percentage

nan_dataset = airquality_df.isnull().mean().sort_values().round(4)*100





# No missing values in DateTime, Date, and Time (-)

#Make float converter function

def make_float(i):

    try: #try - except

        return float(i)

    except:

        return pd.np.nan



# apply make_int function to the entire series using map

for column_name, rows_over_column in airquality_df.iteritems(): 

    if column_name != 'index' and column_name != 'Name' and column_name != 'GPS' and column_name != 'DateTime' and column_name != 'Date' and column_name != 'Time':

        airquality_df[column_name].interpolate(method='linear', direction = 'forward', inplace=True) 



nan_dataset2 = airquality_df.isnull().mean().sort_values().round(4)*100



# Finish missing data



## MODELLING PHASE 1 ##

airquality_df.info()



# NOTES #

#Problems with AQI_Bucket --> many missing values (720 rows or 62%)

#Removing null values in AQI_AQI

nan_dataset2

airquality_df['AQI_Bucket'].unique()

#After removing rows contain NA in AQI_Bucket, the rows left are 380 rows

airquality_df2 = airquality_df.dropna(axis = 0, subset = ['AQI_Bucket'])



#To make it easier, change the label in AQI_Bucket to number

#'Poor' = 5, 'Very Poor' = 5, 'Severe' = 4, 'Moderate' = 3, 'Satisfactory' = 2,'Good' = 1

#Change the AQI_Bucket to categorical

airquality_df2['AQI_Bucket'] = pd.Categorical(airquality_df2['AQI_Bucket'])

airquality_df2['AQI_Bucket']

#Get the code for each category and create a new column called AQI_Bucket_code

airquality_df2['AQI_Bucket_code'] = airquality_df2['AQI_Bucket'].cat.codes

airquality_df2['AQI_Bucket_code']



airquality_df2.isnull().mean().sort_values().round(4)*100





# =============================================================================

# We are going to compare the models using F1 Score and recall

# =============================================================================





#START Neural Network

# Change label/sytring into number, so it can be modelled

# The AQI calculation uses 7 measures: PM2.5, PM10, SO2, NOx, NH3, CO and O3



#Decide independent variables, these will be assigned to 'X_variables'

airquality_df2.info()



X_variables = airquality_df2[['PM2.5','PM10','NH3','SO2','NOx','CO','O3']]

X_variables                                  



Y_variables = airquality_df2[['AQI_Bucket_code']]

Y_variables



#Normalizing the data

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

#Y_variables = preprocessing.normalize(Y_variables)



# normalize the data attributes

# encode class values as integers

encoder = LabelEncoder()

encoder.fit(Y_variables)

encoded_Y = encoder.transform(Y_variables)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)





from sklearn.model_selection import train_test_split

#Split the data into 70 for training 30 for testing

X_train, X_test, y_train, y_test = train_test_split(X_variables, dummy_y, test_size=0.30, random_state=42)





print(X_train.shape)

print(X_test.shape)





#Dependencies

from keras.models import Sequential

from keras.layers import Dense

from keras.losses import CategoricalHinge, SquaredHinge, Hinge





# Neural network

model = Sequential()

# Find Hyper parameter

model.add(Dense(10, input_dim=7, activation='relu'))

model.add(Dense(11, activation='sigmoid')) #sigmod -> relu

model.add(Dense(11, activation='sigmoid')) #sigmod -> relu

model.add(Dense(6, activation='softmax'))





# Find Hyper parameter

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

##model.compile(loss='poisson', optimizer='adam', metrics=['accuracy'])

#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#model.compile(loss=Hinge(), optimizer='adam', metrics=['accuracy'])

#model.compile(loss=SquaredHinge(), optimizer='adam', metrics=['accuracy'])

#model.compile(loss=CategoricalHinge(), optimizer='adam', metrics=['accuracy'])



#model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

#model.compile(loss='poisson', optimizer='Nadam', metrics=['accuracy'])

#model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])

#model.compile(loss=Hinge(), optimizer='Nadam', metrics=['accuracy'])

#model.compile(loss=SquaredHinge(), optimizer='Nadam', metrics=['accuracy'])

#model.compile(loss=CategoricalHinge(), optimizer='Nadam', metrics=['accuracy'])







history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), shuffle=True)

y_pred = model.predict(X_test)



#MATRIX FOR EVALUATION 

#For reference in evaluating ML models: https://www.jeremyjordan.me/evaluating-a-machine-learning-model/

#Import from sklearn.metrics



from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

from scipy import interp



#Confusion matrix(y_test, y_predict)

print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))



#Plot accuracy and loss for every epoch

import matplotlib.pyplot as plt

from itertools import cycle

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()





plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()





lw = 2 #line weight

n_classes = dummy_y.shape[1]

########Plot ROC curves for the multilabel problem

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])





# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Some extension of Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()

print("")



# Get ROC AUC Scores

macro_roc_auc = roc_auc_score(y_test, y_pred,  average="macro")

micro_roc_auc = roc_auc_score(y_test, y_pred,  average="micro")

weighted_roc_auc = roc_auc_score(y_test, y_pred, average="weighted")

samples_roc_auc = roc_auc_score(y_test, y_pred, average="samples")

#macro_roc_auc_ovr = roc_auc_score(y_test, y_prob,  average="macro")

#micro_roc_auc_ovr = roc_auc_score(y_test, y_prob,  average="micro")

#weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, average="weighted")

print("ROC AUC scores:\n{:.6f} (macro),\n{:.6f} (micro),\n{:.6f} (weighted by prevalence),\n{:.6f} (samples)".format(macro_roc_auc, micro_roc_auc, weighted_roc_auc, samples_roc_auc))

#print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} (micro),\n{:.6f} (weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr))