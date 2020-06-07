#############################################
# Read in libraries.
#############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#For QQ Plot
import scipy.stats as sts

#Correlation p-values
from scipy.stats.stats import pearsonr

#####################################################
# Set up working directory.
#####################################################

os.getcwd()
os.chdir('C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Data for Tutorials and ICE\\Data')
os.getcwd()

#############################################
# Read in data.
#############################################

orig_reduc_data = pd.read_table('reduction_data_new.txt', sep = '\t')
orig_reduc_data.columns
orig_reduc_data.dtypes
reduc_data = orig_reduc_data.ix[:,['peruse03','pereou03','intent01','operatingsys','educ_level','eatout']]
reduc_data.columns
reduc_data = reduc_data.dropna()

#################################################
# Descriptive analysis.
#################################################

######################################################################
# Describe data (mean, standard deviation, etc.). Assess linearity.	
######################################################################

#### Boxplot Variables
reduc_data.boxplot()

#### peruse03 (IV)
reduc_data.peruse03.describe()					        #Obtain descriptives
reduc_data.peruse03.plot()                              #Index plot
reduc_data.plot.scatter(x='peruse03', y='intent01')     #Check for linearity

#### pereou03 (IV)
reduc_data.pereou03.describe()					        #Obtain descriptives
reduc_data.pereou03.plot()                              #Index plot
reduc_data.plot.scatter(x='pereou03', y='intent01')     #Check for linearity

#### operatingsys (IV)
reduc_data.operatingsys.describe()					    #Obtain descriptives
reduc_data.operatingsys.plot()                          #Index plot
reduc_data.plot.scatter(x='operatingsys', y='intent01') #Check for linearity

#### educ_level (IV)
reduc_data.educ_level.describe()					    #Obtain descriptives
reduc_data.educ_level.plot()                            #Index plot
reduc_data.plot.scatter(x='educ_level', y='intent01')   #Check for linearity

#### eatout (IV)
reduc_data.eatout.describe()					         #Obtain descriptives
reduc_data.eatout.plot()                                 #Index plot
reduc_data.plot.scatter(x='eatout', y='intent01')        #Check for linearity

#### Intent01 (DV)
reduc_data.intent01.describe()					        #Obtain descriptives
plot_leverage_resid2(linreg2)