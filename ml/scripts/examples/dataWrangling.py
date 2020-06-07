#Import libraries.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

#Set working directory and confirm change in directory.
workingdirectory = 'C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\OOP for Data Science\\Data for Tutorials and ICE\\Data'
os.chdir(workingdirectory)
os.getcwd()

#Declare tables to join, join them, and output result.
leftTable = pd.read_table('CaliforniaHospitalData.csv', sep = ",")
rightTable = pd.read_table('CaliforniaHospitalData_Personnel.txt', sep = "\t")
combinedHospitalData = pd.concat([leftTable, rightTable], axis=1, join = 'inner')
combinedHospitalData


#Remove duplicate columns, Work_ID, Position_ID, and Website. Disply changes in data frame
combinedHospitalData.drop(['HospitalID','Work_ID','PositionID','Website'], axis = 1, inplace = True)
combinedHospitalData

#Select hospitals that are "Small/Rural", have 15 or more available beds, and positive operating income. Store in new data frame.
#Export new data frame as a tab delimited text file entitled hospital_data_new.txt
hospitalDataNew = combinedHospitalData[(combinedHospitalData.Teaching == 'Small/Rural') & (combinedHospitalData.AvlBeds >= 15) & (combinedHospitalData.OperInc >= 0)]
hospitalDataNew
hospitalDataNew.to_csv(r'hospital_data_new.txt', index = None, sep='\t')

#Import hospital_data_new.txt and change column names. Display data in data frame.
hospitalDataNew1 = pd.read_table('hospital_data_new.txt', sep = "\t")
hospitalDataNew1.rename(columns = {'NoFTE':'FullTimeCount'}, inplace = True)
hospitalDataNew1.rename(columns = {'NetPatRev':'NetPatientRevenue'}, inplace = True)
hospitalDataNew1.rename(columns = {'InOperExp':'InpatientOperExp'}, inplace = True)
hospitalDataNew1.rename(columns = {'OutOperExp':'OutpatientOperExp'}, inplace = True)
hospitalDataNew1.rename(columns = {'OperRev':'Operating_Revenue'}, inplace = True)
hospitalDataNew1.rename(columns = {'OperInc':'Operating_Income'}, inplace = True)
hospitalDataNew1

#Overwrite first two rows of data. 
# First row...
hospitalDataNew1.ix[0,15] = 'Acting Director'
hospitalDataNew1.ix[0,12] = 'McCallum'
hospitalDataNew1.ix[0,13] = 'Jonathan'
hospitalDataNew1.ix[0,18] = '1/26/2017'
hospitalDataNew1.ix[0,16] = 248904
hospitalDataNew1.ix[0,17] = 8
hospitalDataNew1.ix[0,0] = 'OU Medical'
hospitalDataNew1.ix[0,1] = 73034
hospitalDataNew1.ix[0,2] = 'Investor'
hospitalDataNew1.ix[0,3] = 'Teaching'
hospitalDataNew1.ix[0,4] = 'Alumni'
hospitalDataNew1.ix[0,5] = 401
hospitalDataNew1.ix[0,6] = 139171.3798
hospitalDataNew1.ix[0,7] = 23385571.1
hospitalDataNew1.ix[0,8] = 24661356.9
hospitalDataNew1.ix[0,9] = 51087342
hospitalDataNew1.ix[0,10] = 3040416
hospitalDataNew1.ix[0,11] = 56
hospitalDataNew1.ix[0,14] = 'F'
#Second row...
hospitalDataNew1.ix[1,15] = 'Safety Inspection Member'
hospitalDataNew1.ix[1,12] = 'McCallum'
hospitalDataNew1.ix[1,13] = 'Jonathan'
hospitalDataNew1.ix[1,18] = '1/26/2017'
hospitalDataNew1.ix[1,16] = 23987
hospitalDataNew1.ix[1,17] = 2
hospitalDataNew1.ix[1,0] = 'Mercy Hospital'
hospitalDataNew1.ix[1,1] = 12345
hospitalDataNew1.ix[1,2] = 'District'
hospitalDataNew1.ix[1,3] = 'Teaching'
hospitalDataNew1.ix[1,4] = 'Alumni'
hospitalDataNew1.ix[1,5] = 263
hospitalDataNew1.ix[1,6] = 116798.8306
hospitalDataNew1.ix[1,7] = 13684503.49
hospitalDataNew1.ix[1,8] = 15159987.51
hospitalDataNew1.ix[1,9] = 42845643
hospitalDataNew1.ix[1,10] = 14001154
hospitalDataNew1.ix[1,11] = 43
hospitalDataNew1.ix[1,14] = 'F'
hospitalDataNew1

#Convert date-time columns into datetime datatypes.
hospitalDataNew1['StartDate'] = pd.to_datetime(hospitalDataNew1['StartDate'])
hospitalDataNew1.dtypes

#Select all the Regional Representatives whose employer's operating income is greater than $100,000. Save data in new data frame. Export data to new text file. Display data.
hospitalDataNew2 = hospitalDataNew1[(hospitalDataNew1.PositionTitle == 'Regional Representative') & (hospitalDataNew1.Operating_Income > 100000)]
hospitalDataNew2.to_csv(r'hospital_data_new_pt2.txt', index = None, sep='\t')
hospitalDataNew2

#Select non-profit hospitals with more than 250 employees and net patient revenue greater than or equal to $109,000. Save data in new data frame. Export data to new text file.
hospitalDataNew3 = combinedHospitalData[((combinedHospitalData.TypeControl == 'Non Profit') & (combinedHospitalData.NetPatRev >= 109000) & (combinedHospitalData.NoFTE > 250))]
hospitalDataNew3.drop(hospitalDataNew3.columns[[12,13,14,15,16,17,18]], axis = 1, inplace = True)
hospitalDataNew3
hospitalDataNew3.to_csv(r'hospital_data_new_pt3.txt', index = None, sep='\t')

#Create training and testing dataset by using a k-fold cross validation technique. Export both sets to csv.
kf = KFold(len(combinedHospitalData.index), n_folds = 4)
for train, test in kf:
    print("%s %s" % (train, test))
combinedHospitalData.ix[train].to_csv(r'training_data.csv', index = None, sep = ',')
combinedHospitalData.ix[test].to_csv(r'testing_data.csv', index = None, sep = ',')



