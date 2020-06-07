################################
#Import libraries.
################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from sklearn import metrics

#For displaying the tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

################################
#Change directory.
################################
os.getcwd()
os.chdir('C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Data for Tutorials and ICE\\Data')
os.getcwd()

###################################################
#Read in data, manipulate data into new dataframes, 
#and display descriptive statistics.
###################################################
###Read, manipulate, and describe descriptive stats for reduction_data_new.txt
orig_data_reduc = pd.read_table('reduction_data_new.txt', sep='\t')
reduc_data = orig_data_reduc.ix[:,['intent01', 'time', 'peruse01', 'peruse02', 'peruse03', 'peruse04', 'peruse05',
       'peruse06', 'pereou01', 'pereou02', 'pereou03', 'pereou04', 'pereou05',
       'pereou06', 'intent02', 'intent03', 'operatingsys',
       'gender', 'educ_level', 'race_white', 'race_black', 'race_hisp',
       'race_asian', 'race_native', 'race_pacif', 'race_other', 'age',
       'citizenship', 'state', 'military', 'familystruct',
       'income', 'employ', 'color', 'eatout', 'religion']]
reduc_data = reduc_data.dropna()
reduc_data.dtypes
reduc_data.columns
len(reduc_data.index)
len(reduc_data.columns)
reduc_data.head()
reduc_data.describe()

###Read, manipulate, and describe descriptive stats for titanic_data.txt
orig_data_titanic = pd.read_table('titanic_data.txt', sep='\t')
titanic_data = orig_data_titanic.ix[:,['Survived','Class', 'Sex', 'Age']]
titanic_data = titanic_data.dropna()
titanic_data.dtypes
titanic_data['Class'] = titanic_data['Class'].map({'1st': 1, '2nd': 2, '3rd':3, 'Crew':4})
titanic_data['Sex'] = titanic_data['Sex'].map({'Female': 2, 'Male': 1})
titanic_data['Age'] = titanic_data['Age'].map({'Adult': 2, 'Child': 1})
titanic_data['Survived'] = titanic_data['Survived'].astype('category')        #convert to categorical dtype
titanic_data['Class'] = titanic_data['Class'].astype('category')              #convert to categorical dtype
titanic_data['Sex'] = titanic_data['Sex'].astype('category')                  #convert to categorical dtype
titanic_data['Age'] = titanic_data['Age'].astype('category')                  #convert to categorical dtype
titanic_data.columns
len(titanic_data.index)
len(titanic_data.columns)
titanic_data.head()
titanic_data.describe(include=['category'])

#########################################################
#Create regression tree.
#Intent01 is a continuous variable.	
#Intent01 is the target	variable.						            
#########################################################
###Original regression tree.
col_names = list(reduc_data.ix[:,1:37].columns.values)
tre1 = tree.DecisionTreeRegressor().fit(reduc_data.ix[:,1:37],reduc_data.intent01)

dot_data = StringIO()
tree.export_graphviz(tre1, out_file=dot_data,
                     feature_names=col_names,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

##################################################################
#Original tree resulted in a large tree that is difficult to use.
#Use the criteria min_samples_split and min_samples_leaf
#and set both to 20.					            
##################################################################
###Pruned tree. 
col_names = list(reduc_data.ix[:,1:37].columns.values)
tre1 = tree.DecisionTreeRegressor(min_samples_split=20,min_samples_leaf=20)
tre1.fit(reduc_data.ix[:,1:37],reduc_data.intent01)

dot_data = StringIO()
tree.export_graphviz(tre1, out_file=dot_data,
                     feature_names=col_names,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


#################################################
# This tree is a classification tree using	  
# categorical independent variables.		 
#################################################
###Full classification tree.
col_names = list(titanic_data.columns.values)
classnames = list(titanic_data.Survived.unique()) #Makes the tree 
tre2 = tree.DecisionTreeClassifier().fit(titanic_data.ix[:,1:4],titanic_data.Survived)

dot_data = StringIO()
tree.export_graphviz(tre2, out_file=dot_data,
                     feature_names=col_names[1:4],
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
###Pruned classification tree.
col_names = list(titanic_data.columns.values)
classnames = list(titanic_data.Survived.unique()) #Makes the tree 
tre2 = tree.DecisionTreeClassifier(min_samples_split=50,min_samples_leaf=50)
tre2.fit(titanic_data.ix[:,1:4],titanic_data.Survived)

dot_data = StringIO()
tree.export_graphviz(tre2, out_file=dot_data,
                     feature_names=col_names[1:4],
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

###Print classification report.
predicted = tre2.predict(titanic_data.ix[:,1:4])
print(metrics.classification_report(titanic_data.Survived, predicted))

###Print confusion matrix.
cm = metrics.confusion_matrix(titanic_data.Survived, predicted)
print(cm)

###Heat map confusion matrix.
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1],['No','Yes'])