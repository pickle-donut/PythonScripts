#############################################
#=============Read in Libraries=============#
#############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import metrics

#Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier


#####################################################
#============Setup the Working Directory============#
#####################################################
os.chdir('C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Data for Tutorials and ICE\\TimeSeriesData')


#############################################
#===============Read in data================#
#############################################
# California financial data.
californiaFinData = pd.read_table('CaliforniaHospital_FinancialData.csv', sep = ',')
californiaFinData
californiaFinData.columns
californiaFinData.dtypes


#############################################################
#=============Time Series Analysis: ARMA (1,1)==============#
# ANN Regression for NET_TOT, TOT_OP_EXP, & NONOP_REV.      #
#############################################################
####################################################################################################################

#==================================
# NET_TOT ANN
#==================================
# Step 1: Copy data.
s1 = californiaFinData.NET_TOT


# Step 2: Create lag effects/time variable.
lagcol = pd.Series([0])
lagcol = lagcol.append(s1, ignore_index = True)
lagcol = lagcol.ix[0:38,]
newcols1 = pd.DataFrame({'lag1': lagcol})


# Step 3: Add data back into dataframe.
# Create time variable.
timelen = len(californiaFinData.index) + 1
newcols2 = pd.DataFrame({'time': list(range(1,timelen))})

californiaFinData_NET_TOT = pd.concat([californiaFinData, newcols1, newcols2], axis=1)


#Finalized data with 1 lag effect.
californiaFinData_NET_TOT = californiaFinData_NET_TOT[['NET_TOT','time','lag1']]


# Data splitting for time series. 
# The data must be linear and incremental.
splitnum = np.round((len(californiaFinData_NET_TOT.index) * 0.7), 0).astype(int)
splitnum

californiaFinData_NET_TOT_train = californiaFinData_NET_TOT.ix[0:27,]
californiaFinData_NET_TOT_test = californiaFinData_NET_TOT.ix[28:38,]


# Neural network code.
nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(californiaFinData_NET_TOT_train[['time','lag1']], californiaFinData_NET_TOT_train.NET_TOT)

nnts1_pred = nnts1.predict(californiaFinData_NET_TOT_test[['time','lag1']])

metrics.mean_absolute_error(californiaFinData_NET_TOT_test.NET_TOT, nnts1_pred)

metrics.mean_squared_error(californiaFinData_NET_TOT_test.NET_TOT, nnts1_pred)


####################################################################################################################

#==================================
# TOT_OP_EXP ANN
#==================================
# Step 1: Copy data.
s2 = californiaFinData.TOT_OP_EXP


# Step 2: Create lag effects/time variable.
lagcol = pd.Series([0,0])
lagcol = lagcol.append(s2, ignore_index = True)
lagcol = lagcol.ix[0:38,]
newcols1 = pd.DataFrame({'lag1': lagcol})


# Step 3: Add data back into dataframe.
# Create time variable.
timelen = len(californiaFinData.index) + 1
newcols2 = pd.DataFrame({'time': list(range(1,timelen))})

californiaFinData_TOT_OP_EXP = pd.concat([californiaFinData, newcols1, newcols2], axis=1)


#Finalized data with 1 lag effect.
californiaFinData_TOT_OP_EXP = californiaFinData_TOT_OP_EXP[['TOT_OP_EXP','time','lag1']]


# Data splitting for time series. 
# The data must be linear and incremental.
splitnum = np.round((len(californiaFinData_TOT_OP_EXP.index) * 0.7), 0).astype(int)
splitnum

californiaFinData_TOT_OP_EXP_train = californiaFinData_TOT_OP_EXP.ix[0:27,]
californiaFinData_TOT_OP_EXP_test = californiaFinData_TOT_OP_EXP.ix[28:38,]


# Neural network code.
nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(californiaFinData_TOT_OP_EXP_train[['time','lag1']], californiaFinData_TOT_OP_EXP_train.TOT_OP_EXP)

nnts1_pred = nnts1.predict(californiaFinData_TOT_OP_EXP_test[['time','lag1']])

metrics.mean_absolute_error(californiaFinData_TOT_OP_EXP_test.TOT_OP_EXP, nnts1_pred)

metrics.mean_squared_error(californiaFinData_TOT_OP_EXP_test.TOT_OP_EXP, nnts1_pred)


####################################################################################################################

#==================================
# NONOP_REV ANN
#==================================
# Step 1: Copy data.
s3 = californiaFinData.NONOP_REV


# Step 2: Create lag effects/time variable.
lagcol = pd.Series([0,0])
lagcol = lagcol.append(s3, ignore_index = True)
lagcol = lagcol.ix[0:38,]
newcols1 = pd.DataFrame({'lag1': lagcol})


# Step 3: Add data back into dataframe.
# Create time variable.
timelen = len(californiaFinData.index) + 1
newcols2 = pd.DataFrame({'time': list(range(1,timelen))})

californiaFinData_NONOP_REV = pd.concat([californiaFinData, newcols1, newcols2], axis=1)


#Finalized data with 1 lag effect.
californiaFinData_NONOP_REV = californiaFinData_NONOP_REV[['NONOP_REV','time','lag1']]


# Data splitting for time series. 
# The data must be linear and incremental.
splitnum = np.round((len(californiaFinData_NONOP_REV.index) * 0.7), 0).astype(int)
splitnum

californiaFinData_NONOP_REV_train = californiaFinData_NONOP_REV.ix[0:27,]
californiaFinData_NONOP_REV_test = californiaFinData_NONOP_REV.ix[28:38,]


# Neural network code.
nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(californiaFinData_NONOP_REV_train[['time','lag1']], californiaFinData_NONOP_REV_train.NONOP_REV)

nnts1_pred = nnts1.predict(californiaFinData_NONOP_REV_test[['time','lag1']])

metrics.mean_absolute_error(californiaFinData_NONOP_REV_test.NONOP_REV, nnts1_pred)

metrics.mean_squared_error(californiaFinData_NONOP_REV_test.NONOP_REV, nnts1_pred)


####################################################################################################################


#############################################################
#=============Time Series Analysis: ARMA (2,0)==============#
# ANN Regression for NET_TOT, TOT_OP_EXP, & NONOP_REV.      #
#############################################################
####################################################################################################################

#==================================
# NET_TOT ANN
#==================================
# Step 1: Copy data.
s1 = californiaFinData.NET_TOT


# Step 2: Create lag effects/time variable.
lagcol = pd.Series([0])
lagcol = lagcol.append(s1, ignore_index = True)
lagcol = lagcol.ix[0:38,]
newcols1 = pd.DataFrame({'lag1': lagcol})

lagcol = pd.Series([0])
lagcol = lagcol.append(s1, ignore_index=True)
lagcol = lagcol.ix[0:38,]
newcols2 = pd.DataFrame({'lag2': lagcol})


# Step 3: Add data back into dataframe.
# Create time variable.
timelen = len(californiaFinData.index) + 1
newcols3 = pd.DataFrame({'time': list(range(1,timelen))})

californiaFinData_NET_TOT = pd.concat([californiaFinData, newcols1, newcols2, newcols3], axis=1)


#Finalized data with 2 lag effects.
californiaFinData_NET_TOT = californiaFinData_NET_TOT[['NET_TOT','time','lag1','lag2']]


# Data splitting for time series. 
# The data must be linear and incremental.
splitnum = np.round((len(californiaFinData_NET_TOT.index) * 0.7), 0).astype(int)
splitnum

californiaFinData_NET_TOT_train = californiaFinData_NET_TOT.ix[0:27,]
californiaFinData_NET_TOT_test = californiaFinData_NET_TOT.ix[28:38,]


# Neural network code.
nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(californiaFinData_NET_TOT_train[['time','lag1','lag2']], californiaFinData_NET_TOT_train.NET_TOT)

nnts1_pred = nnts1.predict(californiaFinData_NET_TOT_test[['time','lag1','lag2']])

metrics.mean_absolute_error(californiaFinData_NET_TOT_test.NET_TOT, nnts1_pred)

metrics.mean_squared_error(californiaFinData_NET_TOT_test.NET_TOT, nnts1_pred)


####################################################################################################################

#==================================
# TOT_OP_EXP ANN
#==================================
# Step 1: Copy data.
s2 = californiaFinData.TOT_OP_EXP


# Step 2: Create lag effects/time variable.
lagcol = pd.Series([0,0])
lagcol = lagcol.append(s2, ignore_index = True)
lagcol = lagcol.ix[0:38,]
newcols1 = pd.DataFrame({'lag1': lagcol})

lagcol = pd.Series([0])
lagcol = lagcol.append(s2, ignore_index=True)
lagcol = lagcol.ix[0:38,]
newcols2 = pd.DataFrame({'lag2': lagcol})


# Step 3: Add data back into dataframe.
# Create time variable.
timelen = len(californiaFinData.index) + 1
newcols3 = pd.DataFrame({'time': list(range(1,timelen))})

californiaFinData_TOT_OP_EXP = pd.concat([californiaFinData, newcols1, newcols2, newcols3], axis=1)


# Finalized data with 2 lag effects.
californiaFinData_TOT_OP_EXP = californiaFinData_TOT_OP_EXP[['TOT_OP_EXP','time','lag1','lag2']]


# Data splitting for time series. 
# The data must be linear and incremental.
splitnum = np.round((len(californiaFinData_TOT_OP_EXP.index) * 0.7), 0).astype(int)
splitnum

californiaFinData_TOT_OP_EXP_train = californiaFinData_TOT_OP_EXP.ix[0:27,]
californiaFinData_TOT_OP_EXP_test = californiaFinData_TOT_OP_EXP.ix[28:38,]


# Neural network code.
nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(californiaFinData_TOT_OP_EXP_train[['time','lag1','lag2']], californiaFinData_TOT_OP_EXP_train.TOT_OP_EXP)

nnts1_pred = nnts1.predict(californiaFinData_TOT_OP_EXP_test[['time','lag1','lag2']])

metrics.mean_absolute_error(californiaFinData_TOT_OP_EXP_test.TOT_OP_EXP, nnts1_pred)

metrics.mean_squared_error(californiaFinData_TOT_OP_EXP_test.TOT_OP_EXP, nnts1_pred)


####################################################################################################################

#==================================
# NONOP_REV ANN
#==================================
# Step 1: Copy data.
s3 = californiaFinData.NONOP_REV


# Step 2: Create lag effects/time variable.
lagcol = pd.Series([0,0])
lagcol = lagcol.append(s3, ignore_index = True)
lagcol = lagcol.ix[0:38,]
newcols2 = pd.DataFrame({'lag1': lagcol})

lagcol = pd.Series([0,0])
lagcol = lagcol.append(s2, ignore_index=True)
lagcol = lagcol.ix[0:38,]
newcols2 = pd.DataFrame({'lag2': lagcol})


# Step 3: Add data back into dataframe.
# Create time variable.
timelen = len(californiaFinData.index) + 1
newcols3 = pd.DataFrame({'time': list(range(1,timelen))})

californiaFinData_NONOP_REV = pd.concat([californiaFinData, newcols1, newcols2, newcols3], axis=1)


#Finalized data with 2 lag effects.
californiaFinData_NONOP_REV = californiaFinData_NONOP_REV[['NONOP_REV','time','lag1','lag2']]


# Data splitting for time series. 
# The data must be linear and incremental.
splitnum = np.round((len(californiaFinData_NONOP_REV.index) * 0.7), 0).astype(int)
splitnum

californiaFinData_NONOP_REV_train = californiaFinData_NONOP_REV.ix[0:27,]
californiaFinData_NONOP_REV_test = californiaFinData_NONOP_REV.ix[28:38,]


# Neural network code.
nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(californiaFinData_NONOP_REV_train[['time','lag1','lag2']], californiaFinData_NONOP_REV_train.NONOP_REV)

nnts1_pred = nnts1.predict(californiaFinData_NONOP_REV_test[['time','lag1','lag2']])

metrics.mean_absolute_error(californiaFinData_NONOP_REV_test.NONOP_REV, nnts1_pred)

metrics.mean_squared_error(californiaFinData_NONOP_REV_test.NONOP_REV, nnts1_pred)


####################################################################################################################
