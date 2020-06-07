#Read in libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Clustering modules
import sklearn.metrics as metcs
from scipy.cluster import hierarchy as hier
from sklearn import cluster as cls

#For the tree
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
import pydotplus


#Change woring directory.
os.getcwd()
os.chdir('C:\\Users\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Data for Tutorials and ICE\\Data')
os.getcwd()

#Read in data.
kmeans_data = pd.read_table('car.test.frame.txt')
kmeans_data['Type'] = kmeans_data['Type'].astype('category')
kmeans_data.columns
kmeans_data.dtypes
kmeans_data.head()
kmeans_data.shape
kmeans_data.describe()

# Perform Cluster Analysis.
kmeans_data.Type.unique()

#Use 6 clusters.
km = cls.KMeans(n_clusters=6).fit(kmeans_data.loc[:,['Mileage','Price']])
km.labels_      

#Use 8 clusters.
km2 = cls.KMeans(n_clusters=8).fit(kmeans_data.loc[:,['Mileage','Price']])
km2.labels_  

#Perform agglomerative clustering
agg1 = cls.AgglomerativeClustering(linkage='ward').fit(kmeans_data[['Mileage','Price']])
agg1.labels_

#Create a plot to view the output
z = hier.linkage(kmeans_data[['Mileage', 'Price']], 'single')
plt.figure()
dn = hier.dendrogram(z)
