
#4. For a given GDP threshold (upper limit), how much is the average percentage of under 18 migrants among all countries?
#datasets: migrants.csv and 2014_world_gdp_with_codes.csv

#input:
 #   GDP threshold (float)    
#output:
 #   average percentage of under 18 migrants in percent (float)


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
