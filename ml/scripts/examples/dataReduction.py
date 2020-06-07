###############################################################################################################################################
###Read in libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA as pca
from sklearn.decomposition import FactorAnalysis as fact

#Clustering modules
import sklearn.metrics as metcs
from scipy.cluster import hierarchy as hier
from sklearn import cluster as cls

###Change directory.
os.getcwd()
os.chdir('C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Data for Tutorials and ICE\\Data')
os.getcwd()

###Read in the data.
reduc_data = pd.read_table('Mccallum_Jonathan_export.txt', sep = "\t")
reduc_data.drop(['StartDate.1'], axis=1, inplace=True)
reduc_data.columns
len(reduc_data.columns)

###Inserting two new rows of data.
#First Row...
reduc_data.ix[0,0] = 'Mercy Hospital'
reduc_data.ix[0,1] = 12345
reduc_data.ix[0,2] = 'District'
reduc_data.ix[0,3] = 'Teaching'
reduc_data.ix[0,4] = 'Alumni'
reduc_data.ix[0,5] = 263
reduc_data.ix[0,6] = 116798.8306
reduc_data.ix[0,7] = 13684503.49
reduc_data.ix[0,8] = 15159987.51
reduc_data.ix[0,9] = 42845643
reduc_data.ix[0,10] = 14001154
reduc_data.ix[0,11] = 43
reduc_data.ix[0,12] = 'McCallum'
reduc_data.ix[0,13] = 'Jonathan'
reduc_data.ix[0,14] = 'F'
reduc_data.ix[0,15] =   2
reduc_data.ix[0,16] = 'Acting Director'
reduc_data.ix[0,17] = 248904
reduc_data.ix[0,18] = 8
reduc_data.ix[0,19] = '2/8/2017'

#Second row...
reduc_data.ix[1,0] = 'OU Medical'
reduc_data.ix[1,1] = 73034
reduc_data.ix[1,2] = 'Investor'
reduc_data.ix[1,3] = 'Teaching'
reduc_data.ix[1,4] = 'Alumni'
reduc_data.ix[1,5] = 401
reduc_data.ix[1,6] = 139171.3798
reduc_data.ix[1,7] = 23385571.1
reduc_data.ix[1,8] = 24661356.9
reduc_data.ix[1,9] = 51087342
reduc_data.ix[1,10] = 3040416
reduc_data.ix[1,11] = 56
reduc_data.ix[1,12] = 'McCallum'
reduc_data.ix[1,13] = 'Jonathan'
reduc_data.ix[1,14] = 'F'
reduc_data.ix[1,15] =   4
reduc_data.ix[1,16] = 'Safety Inspection Member'
reduc_data.ix[1,17] = 23987
reduc_data.ix[1,18] = 2
reduc_data.ix[1,19] = '2/8/2017'
reduc_data

###View and convert datatypes. Describe data.
reduc_data.dtypes
reduc_data.head()
reduc_data['TypeControl'] = reduc_data['TypeControl'].astype('category')        #convert to categorical dtype
reduc_data['Teaching'] = reduc_data['Teaching'].astype('category')              #convert to categorical dtype
reduc_data['DonorType'] = reduc_data['DonorType'].astype('category')            #convert to categorical dtype
reduc_data['Gender'] = reduc_data['Gender'].astype('category')                  #convert to categorical dtype
reduc_data['PositionID'] = reduc_data['PositionID'].astype('category')          #convert to categorical dtype
reduc_data['PositionTitle'] = reduc_data['PositionTitle'].astype('category')    #convert to categorical dtype
reduc_data['Compensation'] = reduc_data['Compensation'].astype('category')      #convert to categorical dtype
reduc_data['MaxTerm'] = reduc_data['MaxTerm'].astype('category')                #convert to categorical dtype
reduc_data['StartDate'] = pd.to_datetime(reduc_data['StartDate'])               #convert to datetime dtype
rows, cols = reduc_data.shape
rows                                                                            #number of records
cols                                                                            #number of variables
reduc_data.describe()
reduc_data.describe(include=['category'])
reduc_data.MaxTerm.unique()
reduc_data.Compensation.unique()
reduc_data.PositionID.unique()

###Create subsamples for analysis.
reduc_data_pca = reduc_data.sample(frac = 0.5, replace = False)
len(reduc_data_pca.index)
reduc_data_fa = reduc_data.sample(frac = 0.5, replace = False)
len(reduc_data_fa.index)
reduc_data_pca
reduc_data_fa
reduc_data_pca.var()
reduc_data_fa.var()


###Conduct Principle Component Analysis.
reduc_data_pca_tst = reduc_data_pca[['NoFTE', 'NetPatRev', 'InOperExp', 'OutOperExp','OperRev','OperInc', 'AvlBeds']] #Declare new dataframe with only numeric columns for PCA.
pca_result = pca(n_components = 7).fit(reduc_data_pca_tst) #Run PCA.
pca_result.explained_variance_ #Obtain eigenvalues.

###Scree plot.
plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5,6,7], pca_result.explained_variance_ratio_, '-o')
plt.ylabel('Proportion of Variance Explained') 
plt.xlabel('Principal Component') 
plt.xlim(0.75,4.25) 
plt.ylim(0,1.05) 
plt.xticks([1,2,3,4,5,6,7])

###Conduct Factor Analysis with six components indicated in PCA.
reduc_data_fa_tst = reduc_data_fa[['NoFTE', 'NetPatRev', 'InOperExp', 'OutOperExp','OperRev','OperInc', 'AvlBeds']] #Declare new dataframe using columns from PCA for FA.
fact_result = fact(n_components=6).fit(reduc_data_fa_tst)
fact_result.components_

###Conduct Factor Analysis with +1 components indicated in PCA.
reduc_data_fa_tst = reduc_data_fa[['NoFTE', 'NetPatRev', 'InOperExp', 'OutOperExp','OperRev','OperInc', 'AvlBeds']] #Declare new dataframe using columns from PCA for FA.
fact_result = fact(n_components=7).fit(reduc_data_fa_tst)
fact_result.components_

###Conduct Factor Analysis with -1 components indicated in PCA.
reduc_data_fa_tst = reduc_data_fa[['NoFTE', 'NetPatRev', 'InOperExp', 'OutOperExp','OperRev','OperInc', 'AvlBeds']] #Declare new dataframe using columns from PCA for FA.
fact_result = fact(n_components=5).fit(reduc_data_fa_tst)
fact_result.components_

###############################################################################################################################################
###Read in data and conduct a K-Means clustering.
kmeans_data = pd.read_table('Mccallum_Jonathan_export.txt', sep = "\t")

###View and convert datatypes.
kmeans_data.dtypes
kmeans_data['TypeControl'] = kmeans_data['TypeControl'].astype('category')        #convert to categorical dtype
kmeans_data['Teaching'] = kmeans_data['Teaching'].astype('category')              #convert to categorical dtype
kmeans_data['DonorType'] = kmeans_data['DonorType'].astype('category')            #convert to categorical dtype
kmeans_data['Gender'] = kmeans_data['Gender'].astype('category')                  #convert to categorical dtype
kmeans_data['PositionID'] = kmeans_data['PositionID'].astype('category')          #convert to categorical dtype
kmeans_data['PositionTitle'] = kmeans_data['PositionTitle'].astype('category')    #convert to categorical dtype
kmeans_data['Compensation'] = kmeans_data['Compensation'].astype('category')      #convert to categorical dtype
kmeans_data['MaxTerm'] = kmeans_data['MaxTerm'].astype('category')                #convert to categorical dtype
kmeans_data['StartDate'] = pd.to_datetime(kmeans_data['StartDate'])               #convert to datetime dtypes
kmeans_data.MaxTerm.unique()
kmeans_data.PositionID.unique()
kmeans_data.Compensation.unique()

#Use 4 clusters.
len(kmeans_data.index) #Returns 61
km = cls.KMeans(n_clusters = 4).fit(kmeans_data.loc[:,['NoFTE', 'NetPatRev', 'InOperExp', 'OutOperExp','OperRev','OperInc', 'AvlBeds']])
km.labels_      #assigned clusters

#Create a confusion matrix
cm1 = metcs.confusion_matrix(kmeans_data.MaxTerm, km.labels_)
print(cm1)       #Printed matrix

#Create a confusion matrix
cm2 = metcs.confusion_matrix(kmeans_data.Compensation, km.labels_)
print(cm2)       #Printed matrix

#Create a confusion matrix
cm3 = metcs.confusion_matrix(kmeans_data.PositionID, km.labels_)
print(cm3)       #Printed matrix


