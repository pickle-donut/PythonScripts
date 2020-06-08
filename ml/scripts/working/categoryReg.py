
#############################################
#=============Read in Libraries=============#
# Read in the necessary libraries.          #
#############################################

import pandas as pd
import numpy as np
import matplotlib as plt
import os

#For QQ Plot
import scipy.stats as sts

#####################################################
#============Setup the Working Directory============#
# Set the working directory to the project folder by#
# running the appropriate code below.               #
#####################################################

os.getcwd()

os.chdir('C:\\Users\\jonmc\Documents\\git\\PythonScripts\\ml\data\\general')
os.getcwd()

red_data = pd.read_table('ProjectDataConsolidated.csv', sep=',')
#==========================
# Perform binning for age
#==========================
red_data['age'].max()
# Max: 48

red_data['age'].min()
# Min: 18
# A total of 31 years

red_data['age'].plot.hist(alpha=0.5)
# The data is skewed, with a greater number
# of individuals with an age closer to 20

# Create non-overlapping sub-intervals (i.e. the bins).
# Select an arbitrary number to start out. 6 bins total
bin_counts,bin_edges,binnum = binned_statistic(red_data['age'], 
                                               red_data['age'], 
                                               statistic='count', 
                                               bins=6)

# Counts within each bin
bin_counts

# Bin Values (only shows left value, not right)
bin_edges

# Result: Due to the skewness of the data, the two bins
# on the left hold the most data. This is not an even
# distribution of the data. Perhaps bin by 2 years, not 5

bin_counts,bin_edges,binnum = binned_statistic(red_data['age'], 
                                               red_data['age'], 
                                               statistic='count', 
                                               bins=15)

bin_counts
# Unlike R, the last bin actually includes the value of 48

bin_edges

#### Results: The first four bins contain the majority of 
#### the data. Take the last twelve bins and combine them
#### into a single bin.

bin_interval = [18, 20, 22, 24, 26, 50]

bin_counts, bin_edges, binnum = binned_statistic(red_data['age'], 
                                                 red_data['age'], 
                                                 statistic='count', 
                                                 bins=bin_interval)

bin_counts

bin_edges

# Recode the values in the age column based on the binning
binlabels = ['age_18_19', 'age_20_21', 'age_22_23', 'age_24_25', 'age_26_48']
age_categ = pd.cut(red_data['age'], bin_interval, right=False, retbins=False, labels=binlabels)

age_categ.name = 'age_categ'

# Take the binning data and add it as a column to the dataframe
red_data = red_data.join(pd.DataFrame(age_categ))

# Compare the original age column to the age_categ
red_data[['age', 'age_categ']].sort_values(by='age')
