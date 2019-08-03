


import zipfile
import numpy as np
import pylab as plt
import pandas as pd


df = pd.read_csv('countries_2.csv')
df.drop(['country','Unnamed: 0'], axis=1, inplace=True)


listC=list(df.columns.values)


listC=list(df.columns.values)
ZM=[]
for i in listC:
    df2005=df.loc[(df['Year'] == 2005)]
    df2017=df.loc[(df['Year'] == 2017)]
    if i != 'Year':
        import2017 = df2017[i].tolist()
        import2005 = df2005[i].tolist()
        Mimport2005=np.mean(import2005)
        Mimport2017=np.mean(import2017)
        Simport2005=np.std(import2005)
        Simport2017=np.std(import2017)
        Nimport2005=len(import2005)
        Nimport2017=len(import2017)
        Z = np.abs((Mimport2005 - Mimport2017)/(np.sqrt(((Simport2005)**2)/(Nimport2005))+(((Simport2017)**2)/(Nimport2017))))
        if Z > 0.05:
            ZM.append(i)
print(ZM)
