#Import libraries.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts #For QQ Plot

#Set working directory and confirm change in directory.
workingdirectory = 'C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\OOP for Data Science\\Data for Tutorials and ICE\\Data'
os.chdir(workingdirectory)
os.getcwd()

#Declare tables to join, join them, and output result.
polluteData = pd.read_table('pollute.txt', sep = "\t")
polluteData.columns
polluteData.rename(columns = {'Wet.days':'Wet_Days'}, inplace = True)

#Get descriptive statistics (Median = 50%), varianace, 
#number of null values, number of unique values, and datatypes for all columns in the table.
polluteData.describe()

polluteData.Pollution.var() #Var = 550.948
polluteData.Temp.var() #Var = 52.240
polluteData.Industry.var() #Var = 317170.988
polluteData.Population.var() #Var = 335371.894
polluteData.Wind.var() #Var = 2.027
polluteData.Rain.var() #Var = 138.579
polluteData.Wet_Days.var() #Var = 702.590

len(polluteData.Pollution.isnull()) #Returns 41
len(polluteData.Temp.isnull()) #Returns 41
len(polluteData.Industry.isnull()) #Returns 41
len(polluteData.Population.isnull()) #Returns 41
len(polluteData.Wind.isnull()) #Returns 41
len(polluteData.Rain.isnull()) #Returns 41
len(polluteData.Wet_Days.isnull()) #Returns 41

len(pd.unique(polluteData.Pollution)) #27 unique values
len(pd.unique(polluteData.Temp)) #39 unique values
len(pd.unique(polluteData.Industry)) #41 unique values
len(pd.unique(polluteData.Population)) #41 unique values
len(pd.unique(polluteData.Wind)) #30 unique values
len(pd.unique(polluteData.Rain)) #41 unique values
len(pd.unique(polluteData.Wet_Days)) #34 unique values

polluteData.dtypes

#Plots to accompany descriptive statistics. 
#Boxplot was chosen to assess outliers; scatterplots for linear trends.
polluteData.boxplot()

polluteData.plot.scatter(x='Pollution', y='Temp', color='DarkBlue', label='Pollution')
polluteData.plot.scatter(x='Pollution', y='Industry', color='DarkBlue', label='Pollution')
polluteData.plot.scatter(x='Pollution', y='Population', color='DarkBlue', label='Pollution')
polluteData.plot.scatter(x='Pollution', y='Wind', color='DarkBlue', label='Pollution')
polluteData.plot.scatter(x='Pollution', y='Rain', color='DarkBlue', label='Pollution')
polluteData.plot.scatter(x='Pollution', y='Wet_Days', color='DarkBlue', label='Pollution')
polluteData.plot.scatter(x='Temp', y='Industry', color='DarkBlue', label='Temp')
polluteData.plot.scatter(x='Temp', y='Population', color='DarkBlue', label='Temp')
polluteData.plot.scatter(x='Temp', y='Wind', color='DarkBlue', label='Temp')
polluteData.plot.scatter(x='Temp', y='Rain', color='DarkBlue', label='Temp')
polluteData.plot.scatter(x='Temp', y='Wet_Days', color='DarkBlue', label='Temp')
polluteData.plot.scatter(x='Industry', y='Population', color='DarkBlue', label='Industry')
polluteData.plot.scatter(x='Industry', y='Wind', color='DarkBlue', label='Industry')
polluteData.plot.scatter(x='Industry', y='Rain', color='DarkBlue', label='Industry')
polluteData.plot.scatter(x='Industry', y='Wet_Days', color='DarkBlue', label='Industry')
polluteData.plot.scatter(x='Population', y='Wind', color='DarkBlue', label='Population')
polluteData.plot.scatter(x='Population', y='Rain', color='DarkBlue', label='Population')
polluteData.plot.scatter(x='Population', y='Wet_Days', color='DarkBlue', label='Population')
polluteData.plot.scatter(x='Wind', y='Rain', color='DarkBlue', label='Wind')
polluteData.plot.scatter(x='Wind', y='Wet_Days', color='DarkBlue', label='Wind')
polluteData.plot.scatter(x='Rain', y='Wet_Days', color='DarkBlue', label='Rain')

#Read in ozone data file, generate QQ plots for each variable, and assess normaility via Shapiro - Wilk.
ozoneData = pd.read_table('ozone.data.txt', sep = "\t")
ozoneData.columns
sts.probplot(ozoneData.rad, dist="norm", plot=plt)
sts.shapiro(ozoneData.rad)
sts.probplot(ozoneData.temp, dist="norm", plot=plt)
sts.shapiro(ozoneData.temp)
sts.probplot(ozoneData.wind, dist="norm", plot=plt)
sts.shapiro(ozoneData.wind)
sts.probplot(ozoneData.ozone, dist="norm", plot=plt)
sts.shapiro(ozoneData.ozone)

#Assess skewness and kurtosis of variables.
#Draw histograms to compare values above against.
polluteData['Pollution'].plot.hist(alpha=0.5)
polluteData.Pollution.skew()
polluteData.Pollution.kurt()
polluteData['Temp'].plot.hist(alpha=0.5)
polluteData.Temp.skew()
polluteData.Temp.kurt()
polluteData['Industry'].plot.hist(alpha=0.5)
polluteData.Industry.skew()
polluteData.Industry.kurt()
polluteData['Population'].plot.hist(alpha=0.5)
polluteData.Population.skew()
polluteData.Population.kurt()
polluteData['Wind'].plot.hist(alpha=0.5)
polluteData.Wind.skew()
polluteData.Wind.kurt()
polluteData['Rain'].plot.hist(alpha=0.5)
polluteData.Rain.skew()
polluteData.Rain.kurt()
polluteData['Wet_Days'].plot.hist(alpha=0.5)
polluteData.Wet_Days.skew()
polluteData.Wet_Days.kurt()






