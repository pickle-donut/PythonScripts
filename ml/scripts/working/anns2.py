#############################################
#=============Read in Libraries=============#
# Read in the necessary libraries.          #
#############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import metrics

#Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Multiple Regression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


#####################################################
#============Setup the Working Directory============#
# Set the working directory to the project folder by#
# running the appropriate code below.               #
#####################################################

os.chdir('C:\\Users\\jonmc\Documents\\git\\PythonScripts\\ml\data\\general')


#############################################
#===============Read in data================#
# Read in the data for both data sets.	    #
#############################################

#================
# Ozone dataset
#================
ozone_data = pd.read_table('ozone.data.txt')
ozone_data.columns
ozone_data.dtypes

# Split data into training and testing
ozone_data_train, ozone_data_test, y_train, y_test = train_test_split(ozone_data[['rad','wind','temp']], 
                                                                      ozone_data.ozone, 
                                                                      test_size=0.4)

# Standardize the scaling of the variables by
# computing the mean and std to be used for later scaling.
scaler = preprocessing.StandardScaler()
scaler.fit(ozone_data_train)

# Perform the standardization process
ozone_data_train = scaler.transform(ozone_data_train)
ozone_data_test = scaler.transform(ozone_data_test)

#===================
# Taxonomy dataset
#===================
taxon_data = pd.read_table('taxonomy.txt', sep='\t')
taxon_data.columns
taxon_data.dtypes

# Split data into training and testing
taxon_data_train, taxon_data_test, taxon_train, taxon_test = train_test_split(taxon_data.ix[:,1:8], 
                                                                              taxon_data.Taxon, 
                                                                              test_size=0.3)

# Standardize the scaling of the variables by
# computing the mean and std to be used for later scaling.
scaler = preprocessing.StandardScaler()
scaler.fit(taxon_data_train)

# Perform the standardization process
taxon_data_train = scaler.transform(taxon_data_train)
taxon_data_test = scaler.transform(taxon_data_test)

#==========================
# Maine Unemployment Data 
# from Maine 1996 to 2006
#==========================
maine_data = pd.read_table('maine_unemployment.txt', sep = '\t')
maine_data.columns


#################################################
#==============Regression Analysis==============#
# Use a regression MLP on the Ozone data.       #
#################################################

nnreg1 = MLPRegressor(activation='logistic', solver='sgd', 
                      hidden_layer_sizes=(20,20), 
                      early_stopping=True)
nnreg1.fit(ozone_data_train, y_train)

nnpred1 = nnreg1.predict(ozone_data_test)

nnreg1.n_layers_

nnreg1.coefs_

metrics.mean_absolute_error(y_test, nnpred1)

metrics.mean_squared_error(y_test, nnpred1)

metrics.r2_score(y_test, nnpred1)
#or use the following:
nnreg1.score(ozone_data_test, y_test)

#=================================
# Compare to multiple regression
#=================================
linreg1 = LinearRegression(fit_intercept=True, normalize=True)
linreg1.fit(ozone_data_train, y_train)

linpred1 = linreg1.predict(ozone_data_test)

metrics.mean_absolute_error(y_test, linpred1)

metrics.mean_squared_error(y_test, linpred1)

metrics.r2_score(y_test, linpred1)

#=================================
# Neural Network using rectified 
# linear unit function
#=================================
nnreg2 = MLPRegressor(activation='relu', solver='sgd', 
                      early_stopping=True)
nnreg2.fit(ozone_data_train, y_train)

nnpred2 = nnreg2.predict(ozone_data_test)

metrics.mean_absolute_error(y_test, nnpred2)

metrics.mean_squared_error(y_test, nnpred2)

metrics.r2_score(y_test, nnpred2)

#Number of nodes per layer
nnreg2.hidden_layer_sizes
#### Result: 100 nodes in a single layer

#=========================================
# Neural Network: (100, 100) nodes/layer
# and no early stopping
#=========================================
nnreg3 = MLPRegressor(activation='relu', solver='sgd', 
                      hidden_layer_sizes=(100,100),
                      early_stopping=True)
nnreg3.fit(ozone_data_train, y_train)

nnpred3 = nnreg3.predict(ozone_data_test)

metrics.mean_absolute_error(y_test, nnpred3)

metrics.mean_squared_error(y_test, nnpred3)

metrics.r2_score(y_test, nnpred3)

#Number of nodes per layer
nnreg3.hidden_layer_sizes
#### Result: 100 nodes in two layers


#################################################
#=================Classification================#
# Perform a classification MLP on the Taxonomy  #
# data. It has a categorical target.            #
#################################################

#==========================
# Use a logistic function
#==========================
nnclass1 = MLPClassifier(activation='logistic', solver='sgd', 
                         hidden_layer_sizes=(100,100))
nnclass1.fit(taxon_data_train, taxon_train)

nnclass1_pred = nnclass1.predict(taxon_data_test)

cm = metrics.confusion_matrix(taxon_test, nnclass1_pred)
print(cm)

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1,2,3], ['I','II','III','IV'])

print(metrics.classification_report(taxon_test, nnclass1_pred))

#=====================================
# Use rectified linear unit function
#=====================================
nnclass2 = MLPClassifier(activation='relu', solver='sgd',
                         hidden_layer_sizes=(100,100))
nnclass2.fit(taxon_data_train, taxon_train)

nnclass2_pred = nnclass2.predict(taxon_data_test)

cm = metrics.confusion_matrix(taxon_test, nnclass2_pred)
print(cm)

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1,2,3], ['I','II','III','IV'])

print(metrics.classification_report(taxon_test, nnclass2_pred))


#################################################
#=============Time Series Analysis==============#
# Conduct a time series analysis using neural   #
# network regression                            #
# ARMA(2,1) model: add lag 2 and lag 6          #
# Based on ACF and PACF                         #
#################################################

#==============================
# Step 1: Make a copy of data
#==============================
s1 = maine_data.unemploy

#==================================
# Step 2: Create Lag Effects by
# adding in empty data, or zeroes
#==================================
#Lag 2 Effect
lag2col = pd.Series([0,0])
lag2col = lag2col.append(s1, ignore_index=True)
lag2col = lag2col.ix[0:127,]

#Lag 6 Effect
lag6col = pd.Series([0,0,0,0,0,0])
lag6col = lag6col.append(s1, ignore_index=True)
lag6col = lag6col.ix[0:127,]

#=======================================
# Step 3: Add data back into dataframe
#=======================================
newcols1 = pd.DataFrame({'lag2': lag2col})
maine_data2 = pd.concat([maine_data, newcols1], axis=1)

newcols2 = pd.DataFrame({'lag6': lag6col})
maine_data3 = pd.concat([maine_data2, newcols2], axis=1)

maine_data3 = maine_data3[['unemploy','lag2','lag6','month','year']]

#=======================
# Create Time Variable
#=======================
timelen = len(maine_data3.index) + 1
newcols3 = pd.DataFrame({'time': list(range(1,timelen))})
maine_data4 = pd.concat([maine_data3, newcols3], axis=1)

#Finalized data with 2 lag effects
maine_data5 = maine_data4[['unemploy','time','lag2','lag6']]

#=====================================
# Data splitting for time series. Do
# not randomly pull data! The data
# must be linear and incremental.
#=====================================
splitnum = np.round((len(maine_data5.index) * 0.7), 0).astype(int)
splitnum
#### Result: 70% of data includes 90 records

maine_train = maine_data5.ix[0:90,]
maine_test = maine_data5.ix[91:127,]

#======================
# Neural network code
#======================
nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(maine_train[['time','lag2','lag6']], maine_train.unemploy)

nnts1_pred = nnts1.predict(maine_test[['time','lag2','lag6']])

metrics.mean_absolute_error(maine_test.unemploy, nnts1_pred)

metrics.mean_squared_error(maine_test.unemploy, nnts1_pred)