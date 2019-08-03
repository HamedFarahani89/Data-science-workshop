
#3. For a given list of countries, what is the average U5MR (under 5 morality rate) ratio between two genders in last year. (girls/boys) 
#datset: su5m.csv

#input:
 #   list of countries (list)
#output:
 #   average of girls U5MR divided by boys U5MR (float)



import os
import numpy as np
import pylab as plt
import pandas as pd
import sys

df = pd.read_csv('su5m.csv')
df['Result'] = df['f2017']/df['m2017']
df2=df.set_index('country')
# we take list of countries as a file in same directory with following line
list_= pd.read_csv(sys.argv[1])
print(list_)
w=[]
for i in list_:

	print(df2['Result'].loc[i])
	w.append(df2['Result'].loc[i])
a = np.array(w)
print('the mean is=' ,a.mean())



