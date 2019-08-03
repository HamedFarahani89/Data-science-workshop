

import zipfile
import numpy as np
import pylab as plt
import pandas as pd
df = pd.read_csv('countries_2.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)


TotalMean=df.mean().tolist()
TotalStd = df.std().tolist()

Countries = np.unique(df['country'].tolist())
meanC=[]
for i in Countries:
    meanC.append(df.loc[df['country']==i].mean().tolist())


Z=[]
for i in range(len(meanC)):
    for j in range(1,len(meanC[i])):
        Z.append(np.abs((meanC[i][j]-TotalMean[j])/(TotalStd[j]*np.sqrt(319))))




Zinternet=[]
ZImports=[]
ZExports=[]
Zmigrants=[]
ZGDP=[]
ZPPP=[]
Zeducation=[]
for i in range(146):
    Zinternet.append(Z[i*7])
for i in range(146):
    ZImports.append(Z[i*7+1])
for i in range(146):
    ZExports.append(Z[i*7+2])
for i in range(146):
    Zmigrants.append(Z[i*7+3])
for i in range(146):
    ZGDP.append(Z[i*7+4])
for i in range(146):
    ZPPP.append(Z[i*7+5])
for i in range(146):
    Zeducation.append(Z[i*7+6])


print('internet:',Countries[Zinternet.index(max(Zinternet))]);
print('ZImports:',Countries[ZImports.index(max(ZImports))]);
print('Exports:',Countries[ZExports.index(max(ZExports))]);
print('migrants:',Countries[Zmigrants.index(max(Zmigrants))]);
print('GDP:',Countries[ZGDP.index(max(ZGDP))]);
print('PPP:',Countries[ZPPP.index(max(ZPPP))]) ;
print('education:',Countries[Zeducation.index(max(Zeducation))])
