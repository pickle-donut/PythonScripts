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

os.chdir('C:\\Users\\jonmc\Documents\\git\\PythonScripts\\ml\data\\general\\')


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
                      hidden_layer_sizes=(100,100))
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
