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

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Multiple Regression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree

#For displaying the tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


#####################################################
#============Setup the Working Directory============#
#####################################################
os.chdir('C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Data for Tutorials and ICE\\Data')


##############
# Part I     #
######################################################################################################################################################
#############################################
#===============Read in data================#
#############################################
# Reduction dataset.
reduc_data = pd.read_table('reduction_data_new.txt')
reduc_data.columns
reduc_data.dtypes
reduction_data = reduc_data[['peruse03','pereou03','operatingsys','educ_level','eatout','intent01']]
reduction_data = reduction_data.dropna()

# Split data into training and testing
reduc_data_train, reduc_data_test, y_train, y_test = train_test_split(reduction_data[['peruse03','pereou03','operatingsys','educ_level','eatout']], 
                                                                      reduction_data.intent01, 
                                                                      test_size=0.4)

# Standardize the scaling of the variables by
# computing the mean and std to be used for later scaling.
scaler = preprocessing.StandardScaler()
scaler.fit(reduc_data_train)

# Perform the standardization process.
reduc_data_train = scaler.transform(reduc_data_train)
reduc_data_test = scaler.transform(reduc_data_test)


#################################################
#==============Regression Analysis==============#
#################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MLP Regressor model.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nnreg1 = MLPRegressor(activation='logistic', solver='sgd', 
                      hidden_layer_sizes=(20,20), 
                      early_stopping=True)
nnreg1.fit(reduc_data_train, y_train)

nnpred1 = nnreg1.predict(reduc_data_test)

nnreg1.n_layers_

nnreg1.coefs_

metrics.mean_absolute_error(y_test, nnpred1)

metrics.mean_squared_error(y_test, nnpred1)

metrics.r2_score(y_test, nnpred1)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Multiple Linear Regression model.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
linreg1 = LinearRegression(fit_intercept=True, normalize=True)
linreg1.fit(reduc_data_train, y_train)

linpred1 = linreg1.predict(reduc_data_test)

metrics.mean_absolute_error(y_test, linpred1)

metrics.mean_squared_error(y_test, linpred1)

metrics.r2_score(y_test, linpred1)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Neural Network using rectified linear unit function.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nnreg2 = MLPRegressor(activation='relu', solver='sgd', 
                      early_stopping=True)
nnreg2.fit(reduc_data_train, y_train)

nnpred2 = nnreg2.predict(reduc_data_test)

metrics.mean_absolute_error(y_test, nnpred2)

metrics.mean_squared_error(y_test, nnpred2)

metrics.r2_score(y_test, nnpred2)

#Number of nodes per layer
nnreg2.hidden_layer_sizes


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Neural Network: (100, 100) nodes/layer and no early stopping.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nnreg3 = MLPRegressor(activation='relu', solver='sgd', 
                      hidden_layer_sizes=(100,100))
nnreg3.fit(reduc_data_train, y_train)

nnpred3 = nnreg3.predict(reduc_data_test)

metrics.mean_absolute_error(y_test, nnpred3)

metrics.mean_squared_error(y_test, nnpred3)

metrics.r2_score(y_test, nnpred3)


##############
# Part II    #
######################################################################################################################################################
#############################################
#===============Read in data================#
#############################################
# Titanic dataset.
titanic_data = pd.read_table('titanic_data.txt')
titanic_data.columns
titanic_data.dtypes
titanic_data = titanic_data.dropna()
titanic_data['Class'] = titanic_data['Class'].map({'1st': 1, '2nd': 2, '3rd':3, 'Crew':4})
titanic_data['Sex'] = titanic_data['Sex'].map({'Female': 2, 'Male': 1})
titanic_data['Age'] = titanic_data['Age'].map({'Adult': 2, 'Child': 1})
titanic_data['Survived'] = titanic_data['Survived'].astype('category')        #convert to categorical dtype
titanic_data['Class'] = titanic_data['Class'].astype('category')              #convert to categorical dtype
titanic_data['Sex'] = titanic_data['Sex'].astype('category')                  #convert to categorical dtype
titanic_data['Age'] = titanic_data['Age'].astype('category')                  #convert to categorical dtype


# Split data into training and testing
titanic_data_train, titanic_data_test, titanic_train, titanic_test = train_test_split(titanic_data[['Class','Sex','Age']], 
                                                                      titanic_data.Survived, 
                                                                      test_size=0.4)

# Standardize the scaling of the variables by
# computing the mean and std to be used for later scaling.
scaler = preprocessing.StandardScaler()
scaler.fit(titanic_data_train)

# Perform the standardization process.
titanic_data_train = scaler.transform(titanic_data_train)
titanic_data_test = scaler.transform(titanic_data_test)


#################################################
#=================Classification================#
#################################################
#++++++++++++++++++++++++++++++++++++++
# Use a logistic function.
#++++++++++++++++++++++++++++++++++++++
nnclass1 = MLPClassifier(activation='logistic', solver='sgd', 
                         hidden_layer_sizes=(100,100))
nnclass1.fit(titanic_data_train, titanic_train)

nnclass1_pred = nnclass1.predict(titanic_data_test)

cm = metrics.confusion_matrix(titanic_test, nnclass1_pred)
print(cm)

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1], ['No','Yes'])

print(metrics.classification_report(titanic_test, nnclass1_pred))



#################################################
# Classification tree.		 
#################################################
# Full classification tree.
col_names = list(titanic_data.columns.values)
classnames = list(titanic_data.Survived.unique()) #Makes the tree 
tre2 = tree.DecisionTreeClassifier().fit(titanic_data.ix[:,0:3],titanic_data.Survived)

dot_data = StringIO()
tree.export_graphviz(tre2, out_file=dot_data,
                     feature_names=col_names[0:3],
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

########################################################
#Use the criteria min_samples_split and min_samples_leaf
#and set both to 50	 
########################################################
# Pruned classification tree.
col_names = list(titanic_data.columns.values)
classnames = list(titanic_data.Survived.unique()) #Makes the tree 
tre2 = tree.DecisionTreeClassifier(min_samples_split=50,min_samples_leaf=50)
tre2.fit(titanic_data.ix[:,0:3],titanic_data.Survived)

dot_data = StringIO()
tree.export_graphviz(tre2, out_file=dot_data,
                     feature_names=col_names[0:3],
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# Print classification report.
predicted = tre2.predict(titanic_data.ix[:,0:3])

# Print confusion matrix.
cm = metrics.confusion_matrix(titanic_data.Survived, predicted)
print(cm)

# Heat map confusion matrix.
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1],['No','Yes'])

print(metrics.classification_report(titanic_data.Survived, predicted))

