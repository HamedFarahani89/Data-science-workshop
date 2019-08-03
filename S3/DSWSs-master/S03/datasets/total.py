



import numpy as np
import pylab as plt
import pandas as pd
import sys


dftesla = pd.read_csv('tesla-stock-price.csv')
dftesla['avg'] = dftesla[['high', 'low']].mean(axis=1)
y = dftesla['avg'].values
Y=y.tolist()
print(Y)
count=0
# when u want take number from sys u should put 'int' unless it take number as str
x=int(sys.argv[1])
print(x)
# method is we take data two in a row and compare. if specific number is between them, it means the specific number cross the line
for i in range(1,len(Y)-1):
    if Y[i]<=x and Y[i+1]>x:
        count+=1
print(count)


##########



import numpy as np
import pylab as plt
import pandas as pd
import sys
sigearth = pd.read_csv('significant-earthquakes.csv')

# this line count when satiffy with two things in data, first of all find the country that user considered and second of all, the 'Number of significant earthquakes (significant earthquakes)' should be '1'

print(len(sigearth[(sigearth['Entity']==sys.argv[1]) & (sigearth['Number of significant earthquakes (significant earthquakes)']==1)]))

##########




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

##########




import numpy as np
import pylab as plt
import pandas as pd
import sys
gdp = pd.read_csv('2014_world_gdp_with_codes.csv')
mig = pd.read_csv('migrants.csv')
gdp = gdp.rename(columns={'COUNTRY':'country'})
GM = pd.merge(gdp,mig ,how='inner', on=['country'])
GM = GM.rename(columns={'GDP (BILLIONS)':'GDP'})
GM2=GM.set_index('GDP')
GM3=GM2.sort_values('GDP',ascending=0)
x=GM3['under18'].loc[:int(sys.argv[1])].tolist()

y=[]
for i in x:
    y.append(float(i))
print(sum(y)/len(y))

##########
حامد فراهانی
