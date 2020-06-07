#############################################
# Read in libraries.
#############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#K-Fold
from sklearn.cross_validation import KFold

#Binning of data
from scipy.stats import binned_statistic

#Regression output
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


#####################################################
# Set up working directory.
#####################################################
os.getcwd()
os.chdir('C:\\Users\\jonmc\\Desktop\\Gradz School\\Semester # 2\\OOP for Data Science\\Data for Tutorials and ICE\\Data')
os.getcwd()


#############################################
# Read in and scrub data.
#############################################
origHospitalData = pd.read_table('McCallum_Jonathan_export.txt', sep = '\t')
origHospitalData.columns
origHospitalData.dtypes
hospitalData = origHospitalData.dropna()
hospitalData['Compensation'] = origHospitalData['Compensation'].map({23987: 1, 46978: 2, 89473:3, 248904:4})

# Convert data types.
hospitalData['TypeControl'] = hospitalData['TypeControl'].astype('category')
hospitalData['Teaching'] = hospitalData['Teaching'].astype('category')
hospitalData['DonorType'] = hospitalData['DonorType'].astype('category')
hospitalData['Gender'] = hospitalData['Gender'].astype('category')
hospitalData['PositionID'] = hospitalData['PositionID'].astype('category')
hospitalData['PositionTitle'] = hospitalData['PositionTitle'].astype('category')
hospitalData['Compensation'] = hospitalData['Compensation'].astype('category')
hospitalData['MaxTerm'] = hospitalData['MaxTerm'].astype('category')
hospitalData['StartDate'] = pd.to_datetime(hospitalData['StartDate'])
hospitalData['StartDate.1'] = pd.to_datetime(hospitalData['StartDate.1'])
hospitalData.dtypes

# Create subset and rename columns.
hospitalSubData = hospitalData[['OperInc','OperRev','AvlBeds','Teaching', 'DonorType', 'Gender', 'PositionTitle', 'Compensation', 'TypeControl']]
hospitalSubData.columns = ['Operating_Income','Operating_Revenue','Available_Beds','Teaching', 'Donor_Type', 'Gender', 'Position_Title', 'Compensation', 'Type_Control']


#################################################
# Create training, validation, and testing sets.
#################################################
kf = KFold(len(hospitalSubData.index), n_folds=2)
for train, test in kf:
    print("%s %s" % (train, test))


#################################################
# Descriptive analysis.
#################################################
#### Describe training set.
hospitalTrain = hospitalSubData.ix[train]
hospitalTrain.reset_index(inplace=True)
hospitalTrain
hospitalTrain.describe()
hospitalTrain.describe(include=['category'])

##################################################################################################################################################################################

#### Describe testing set.
hospitalTest = hospitalSubData.ix[test]
hospitalTest.reset_index(inplace=True)
hospitalTest
hospitalTest.describe()
hospitalTest.describe(include=['category'])


########################################################################################################
# Create dummy variables for Teaching, DonorType, Gender, PositionTitle, Compensation, and TypeControl.
########################################################################################################
#### Create dummy variables for training set.
# Teaching variable.
hospitalTrain.Teaching.unique()
test_dummy1 = pd.get_dummies(hospitalTrain['Teaching'], prefix = 'Teach')
test_dummy1.head()
test_dummy1.columns = ['Teach','RuralSmall']
hospitalTrain = hospitalTrain.join(test_dummy1)

# Donor_Type variable.
hospitalTrain.Donor_Type.unique()
test_dummy2 = pd.get_dummies(hospitalTrain['Donor_Type'], prefix = 'Donor')
test_dummy2.head()
test_dummy2.columns = ['Donor_Alumni','Donor_Charity']
hospitalTrain = hospitalTrain.join(test_dummy2)

# Gender variable.
hospitalTrain.Gender.unique()
test_dummy3 = pd.get_dummies(hospitalTrain['Gender'], prefix = 'Gender')
test_dummy3.head()
test_dummy3.columns = ['Gender_F','Gender_M']
hospitalTrain = hospitalTrain.join(test_dummy3)

# Position_Title variable.
hospitalTrain.Position_Title.unique()
test_dummy4 = pd.get_dummies(hospitalTrain['Position_Title'], prefix = 'Title')
test_dummy4.head()
test_dummy4.columns = ['Acting_Director', 'Regional_Representative', 'Safety_Inspection_Member', 'State_Board_Representative']
hospitalTrain = hospitalTrain.join(test_dummy4)

# Compensation variable.
hospitalTrain.Compensation.unique()
test_dummy5 = pd.get_dummies(hospitalTrain['Compensation'], prefix = 'Comp')
test_dummy5.head()
hospitalTrain = hospitalTrain.join(test_dummy5)

# Type_Control variable.
hospitalTrain.Type_Control.unique()
test_dummy6 = pd.get_dummies(hospitalTrain['Type_Control'], prefix = 'Ctrl')
test_dummy6.head()
test_dummy6.columns = ['City', 'District', 'Investor', 'Non_Profit']
hospitalTrain = hospitalTrain.join(test_dummy6)
hospitalTrain

##################################################################################################################################################################################

#### Create dummy variables for testing set.
# Teaching variable.
hospitalTest.Teaching.unique()
test_dummy1 = pd.get_dummies(hospitalTest['Teaching'], prefix = 'Teach')
test_dummy1.head()
test_dummy1.columns = ['Teach','RuralSmall']
hospitalTest = hospitalTest.join(test_dummy1)

# Donor_Type variable.
hospitalTest.Donor_Type.unique()
test_dummy2 = pd.get_dummies(hospitalTest['Donor_Type'], prefix = 'Donor')
test_dummy2.head()
test_dummy2.columns = ['Donor_Alumni','Donor_Charity']
hospitalTest = hospitalTest.join(test_dummy2)

# Gender variable.
hospitalTest.Gender.unique()
test_dummy3 = pd.get_dummies(hospitalTest['Gender'], prefix = 'Gender')
test_dummy3.head()
test_dummy3.columns = ['Gender_F','Gender_M']
hospitalTest = hospitalTest.join(test_dummy3)

# Position_Title variable.
hospitalTest.Position_Title.unique()
test_dummy4 = pd.get_dummies(hospitalTest['Position_Title'], prefix = 'Title')
test_dummy4.head()
test_dummy4.columns = ['Acting_Director', 'Regional_Representative', 'Safety_Inspection_Member', 'State_Board_Representative']
hospitalTest = hospitalTest.join(test_dummy4)

# Compensation variable.
hospitalTest.Compensation.unique()
test_dummy5 = pd.get_dummies(hospitalTest['Compensation'], prefix = 'Comp')
test_dummy5.head()
hospitalTest = hospitalTest.join(test_dummy5)

# Type_Control variable.
hospitalTest.Type_Control.unique()
test_dummy6 = pd.get_dummies(hospitalTest['Type_Control'], prefix = 'Ctrl')
test_dummy6.head()
test_dummy6.columns = ['City', 'District', 'Investor', 'Non_Profit']
hospitalTest = hospitalTest.join(test_dummy6)
hospitalTest


#################################################
# Create binning.
#################################################
#### Create binning variable for training set.
# Determine range of values.
hospitalTrain['Available_Beds'].min()
hospitalTrain['Available_Beds'].max()

# Total beds.
hospitalTrain['Available_Beds'].max() - hospitalTrain['Available_Beds'].min() + 1

# Plot histogram.
hospitalTrain['Available_Beds'].plot.hist(alpha=0.5)

# Initial binning; pick an arbitrary number.
bin_counts,bin_edges,binnum = binned_statistic(hospitalTrain['Available_Beds'], hospitalTrain['Available_Beds'], statistic='count', bins=7)
bin_counts
bin_edges

# Refining binning.
# Better, but the data is skewed to the right.
bin_counts,bin_edges,binnum = binned_statistic(hospitalTrain['Available_Beds'], hospitalTrain['Available_Beds'], statistic='count', bins=35)
bin_counts
bin_edges

# Bin by specific interval.
bin_interval = [25, 42, 65, 121, 234, 523, 740]
bin_counts, bin_edges, binnum = binned_statistic(hospitalTrain['Available_Beds'], hospitalTrain['Available_Beds'], statistic='count', bins=bin_interval)
bin_counts
bin_edges

# Recode the values in the age column based on the binning
binlabels = ['beds_25_35', 'beds_42_60', 'beds_65_107', 'beds_121_211', 'beds_234_462', 'beds_523_730']
AvlBeds_categ = pd.cut(hospitalTrain['Available_Beds'], bin_interval, right=False, retbins=False, labels=binlabels)

AvlBeds_categ.name = 'AvlBeds_categ'

# Take the binning data and add it as a column to the dataframe.
hospitalTrain = hospitalTrain.join(pd.DataFrame(AvlBeds_categ))

# Compare the original Available_Beds column to the AvlBeds_categ.
hospitalTrain[['Available_Beds', 'AvlBeds_categ']].sort_values(by='Available_Beds')

# Create indicator variables.
hospitalTrain.AvlBeds_categ.unique()
test_dummy7 = pd.get_dummies(hospitalTrain['AvlBeds_categ'], prefix = 'Avl')
test_dummy7.head()
hospitalTrain = hospitalTrain.join(test_dummy7)

##################################################################################################################################################################################

#### Create binning variable for testing set.
# Determine range of values.
hospitalTest['Available_Beds'].min()
hospitalTest['Available_Beds'].max()

# Total beds.
hospitalTest['Available_Beds'].max() - hospitalTest['Available_Beds'].min() + 1

# Plot histogram.
hospitalTest['Available_Beds'].plot.hist(alpha=0.5)

# Initial binning; pick an arbitrary number.
bin_counts,bin_edges,binnum = binned_statistic(hospitalTest['Available_Beds'], hospitalTest['Available_Beds'], statistic='count', bins=7)
bin_counts
bin_edges

# Refining binning.
# Better, but the data is skewed to the right.
bin_counts,bin_edges,binnum = binned_statistic(hospitalTest['Available_Beds'], hospitalTest['Available_Beds'], statistic='count', bins=35)
bin_counts
bin_edges

# Bin by specific interval.
bin_interval = [12, 20, 35, 76, 363, 658, 920]
bin_counts, bin_edges, binnum = binned_statistic(hospitalTest['Available_Beds'], hospitalTest['Available_Beds'], statistic='count', bins=bin_interval)
bin_counts
bin_edges

# Recode the values in the age column based on the binning
binlabels = ['beds_12_15', 'beds_20_28', 'beds_35_62', 'beds_76_146', 'beds_363_606', 'beds_658_920']
AvlBeds_categ = pd.cut(hospitalTest['Available_Beds'], bin_interval, right=False, retbins=False, labels=binlabels)

AvlBeds_categ.name = 'AvlBeds_categ'

# Take the binning data and add it as a column to the dataframe.
hospitalTest = hospitalTest.join(pd.DataFrame(AvlBeds_categ))

# Compare the original Available_Beds column to the AvlBeds_categ.
hospitalTest[['Available_Beds', 'AvlBeds_categ']].sort_values(by='Available_Beds')

# Create indicator variables.
hospitalTest.AvlBeds_categ.unique()
test_dummy7 = pd.get_dummies(hospitalTest['AvlBeds_categ'], prefix = 'Avl')
test_dummy7.head()
hospitalTest = hospitalTest.join(test_dummy7)


#################################################
# Regression analysis.
#################################################
#Get column names.
hospitalTrain.columns

#### Create regression object for Operating Income in training set.
hospTrain_reg1 = smf.ols('Operating_Income ~ Avl_beds_25_35 + Avl_beds_42_60 + Avl_beds_65_107 + Avl_beds_121_211 + Avl_beds_234_462 + City + District + Non_Profit', hospitalTrain).fit()
hospTrain_reg1.summary()

#### Create regression object for Operating Revenue in training set.
hospTrain_reg1 = smf.ols('Operating_Revenue ~ Avl_beds_25_35 + Avl_beds_42_60 + Avl_beds_65_107 + Avl_beds_121_211 + Avl_beds_234_462 + Donor_Charity + RuralSmall', hospitalTrain).fit()
hospTrain_reg1.summary()

##################################################################################################################################################################################

#Get column names.
hospitalTest.columns

#### Create regression object for Operating Income in testing set.
hospTrain_reg1 = smf.ols('Operating_Income ~ Avl_beds_12_15 +  Avl_beds_20_28 + Avl_beds_35_62 +  Avl_beds_76_146 +  Avl_beds_363_606 + City + District + Non_Profit', hospitalTest).fit()
hospTrain_reg1.summary()

#### Create regression object for Operating Revenue in testing set.
hospTrain_reg1 = smf.ols('Operating_Revenue ~ Avl_beds_12_15 +  Avl_beds_20_28 + Avl_beds_35_62 +  Avl_beds_76_146 +  Avl_beds_363_606 + Donor_Charity + RuralSmall', hospitalTest).fit()
hospTrain_reg1.summary()
