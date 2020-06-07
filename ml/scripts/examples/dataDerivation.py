import os
os.getcwd()
os.chdir('C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Data for Tutorials and ICE\Data')
os.getcwd()

#Read in file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
car_data = pd.read_table('car.test.frame.txt', sep='\t') 

#Here are the headers: Price,Country, Reliability, Mileage, Type, Weight, Disp., HP
car_data.head(0) 

#60 rows
len(car_data.index) 

#8 columns
len(car_data.columns) 

#Country & Type are object datatypes
car_data.dtypes 

#Convert Country and Type to categorical
car_data.Country = car_data.Country.astype('category') 
car_data.Type = car_data.Type.astype('category')

#Find unique values in Country and Type
len(pd.unique(car_data.Type)) #6 unique values
len(pd.unique(car_data.Country)) #8 unique values

#Returns USA
car_data.ix[32,1] 

#Returns Price: 6599, Country: Japan, Reliability: 5, Mileage: 32, Type: Small, Weight: 2440, Disp.: 113, HP: 103
car_data.ix[4,:] 

#These all select the 29th row; 2nd, 3rd, & 4th columns 
car_data.ix[28,1:4]
car_data.ix[28,[1,2,3]]
car_data.ix[28,['Country','Reliabilty','Mileage']]

#These select the 45th row; 3rd & 7th columns
car_data.ix[44,[2,6]]
car_data.ix[28,['Reliabilty','Disp.']]

#creates new dataframe for HP
hp = car_data.ix[:,['HP']]
hp = car_data.HP

#Select compact cars that have a reliability greater than and equal to 2.
car_data[(car_data.Type=='Compact')&(car_data.Reliability>=2)]

