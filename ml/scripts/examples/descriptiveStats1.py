################################
# Import libraries.
################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#For QQ Plot
import scipy.stats as sts 

################################
# Change directory.
################################
os.getcwd()
os.chdir('C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Project\\Data-CSV')
os.getcwd()


####################################################
# Read in data, manipulate data into new dataframes, 
# and display descriptive statistics.
####################################################
#### Read, manipulate, convert, and describe data.
orig_proj_data = pd.read_table('ProjectDataConsolidated.csv', sep=',')
orig_proj_data
orig_proj_data['Report_Date'] = pd.to_datetime(orig_proj_data['Report_Date'])
orig_proj_data.isnull().sum()
proj_data = orig_proj_data.dropna()
proj_data.dtypes
proj_data.columns
proj_data.shape
proj_data.head()


#################################################
# Descriptive analysis.
#################################################
#### Describe project data.
proj_data.reset_index(inplace=True)
proj_data
proj_data.describe()
proj_data.describe(include=['datetime64[ns]'])


######################################################################
# Describe using a series of different plots.
######################################################################
matplotlib.style.use('ggplot')

#### Boxplot.
variables = proj_data.ix[:,['Crude_Price','Gas_Price','Gold_Price']]
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
variables.plot.box(color=color, title = "Boxplot of Variables")


#### Index plot.
variables.plot(title = "Index Plot of Variables", legend = True)                          
proj_data.Crude_Price.plot(title = "Crude Price Index Plot", color ="Red")                            
proj_data.Gold_Price.plot(title = "Gold Price Index Plot", color ="Blue")                             
proj_data.Gas_Price.plot(title = "Gas Price Index Plot", color ="Purple")                          


#### Scatterplots.
proj_data.plot.scatter(x='Crude_Price', y='Gas_Price', title = "Crude vs Gas Price Scatterplot")
proj_data.plot.scatter(x='Gold_Price', y='Gas_Price', title = "Gold vs Gas Price Scatterplot")   
proj_data.plot.scatter(x='Crude_Price', y='Gold_Price', title = "Crude vs Gold Price Scatterplot") 


#### Bar charts.
ax = proj_data[['Crude_Price','Gas_Price']].plot(kind='bar', title ="Price Comparison", figsize=(15, 10), color=['r','g'], legend=True, fontsize=12)
ax.set_xlabel("Crude_Price", fontsize=12)
ax.set_ylabel("Gas_Price", fontsize=12)
plt.show()
ax = proj_data[['Gold_Price','Gas_Price']].plot(kind='bar', title ="Price Comparison", figsize=(15, 10), color=['r','g'], legend=True, fontsize=12)
ax.set_xlabel("Gold_Price", fontsize=12)
ax.set_ylabel("Gas_Price", fontsize=12)
plt.show()
ax = proj_data[['Crude_Price','Gold_Price']].plot(kind='bar', title ="Price Comparison", figsize=(15, 10), color=['r','g'], legend=True, fontsize=12)
ax.set_xlabel("Crude_Price", fontsize=12)
ax.set_ylabel("Gold_Price", fontsize=12)
plt.show()


#### Histograms.
proj_data['Crude_Price'].plot.hist(alpha=0.5, title = "Crude Price Histogram", color = "Red")
proj_data.Crude_Price.skew()
proj_data.Crude_Price.kurt()
proj_data['Gold_Price'].plot.hist(alpha=0.5, title = "Gold Price Histogram", color ="Blue")
proj_data.Gold_Price.skew()
proj_data.Gold_Price.kurt()
proj_data['Gas_Price'].plot.hist(alpha=0.5, title = "Gas Price Histogram", color ="Purple")
proj_data.Gas_Price.skew()
proj_data.Gas_Price.kurt()


#### QQ plots.
sts.probplot(proj_data.Crude_Price, dist="norm", plot=plt)
plt.title("Crude Price Q-Q plot", size=16)
sts.shapiro(proj_data.Crude_Price)
sts.probplot(proj_data.Gold_Price, dist="norm", plot=plt)
plt.title("Gold Price Q-Q plot", size=16)
sts.shapiro(proj_data.Gold_Price)
sts.probplot(proj_data.Gas_Price, dist="norm", plot=plt)
plt.title("Gas Price Q-Q plot", size=16)
sts.shapiro(proj_data.Gas_Price)