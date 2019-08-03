


import os
import numpy as np
import pylab as plt
import pandas as pd
import sys

df = pd.read_csv('su5m.csv')
df['Result'] = df['f2017']/df['m2017']
df2=df.set_index('country')
list_ = sys.argv[1]

list_ave=[]
for i in list_:
     print(df2['Result'].loc[i])
     list_ave.append(df2['Result'].loc[i])

#a = np.array(list_ave)
print('the mean is=' ,a.mean())
