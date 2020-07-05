import pandas as pd
import numpy as np
import random as r 

r.seed(15)

rows = 20

val = []
indx = []

time = np.linspace(0,1,20)
flag = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(rows):
  val.append(r.randint(0, 100))
  indx.append(i)

df = pd.DataFrame([time, val, flag])
df = df.transpose()
df.columns = ['time', 'val', 'flag']

########################################################################

#define list to feed unique keys into
final_data = pd.DataFrame()

Keys = [i for i in pandasDF.Key.unique()]

for key in Keys:  
  data = df.loc[data['Key'] == key]

  flags = data.loc[data['flag'] == 1]

  no_flags = data.loc[data['flag'] == 0]

  sampled_flags = flags.sample(n=(len(no_flags)-len(flags)), replace=True, random_state=42)

  flags_all = pd.concat([flags, sampled_flags])

  prep_data = pd.concat([no_flags, flags_all]).sort_values(by=['time'])

  final_data = pd.concat([prep_data, final_data])

prep_data = prep_data.drop(['index'], axis=1)
prep_data = prep_data.reset_index()

print(prep_data)