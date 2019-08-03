

import zipfile
import numpy as np
import pylab as plt
import pandas as pd
df = pd.read_csv('Credit.csv')
df.drop(['Unnamed: 0' and 'Ethnicity'], axis=1, inplace=True)


X = ['Income' , 'Limit' , 'Rating','Cards' ,  'Age'    , 'Education'   , 'Balance']
df1 = df[df['Married'] == 'Yes']
df2 = df[df['Married'] == 'No']
TotalMean1 = df1.mean().tolist()
TotalMean2 = df2.mean().tolist()
TotalStd1 = df1.std().tolist()
TotalStd2 = df2.std().tolist()
n1 = len(df1)
n2 = len(df2)
ZMarried = []
for i in range(6):
    ZMarried.append(np.abs((TotalMean2[i]-TotalMean1[i])/(np.sqrt(((TotalStd2[i])**2/n2)+((TotalStd1[i])**2/n1)))))

print('In marital status:' , X[ZMarried.index(min(ZMarried))])
#Income  Limit         Rating         Cards            Age             Education        Balance        

X = ['Income' , 'Limit' , 'Rating','Cards' ,  'Age'    , 'Education'   , 'Balance']
df1 = df[df['Student'] == 'Yes']
df2 = df[df['Student'] == 'No']
STotalMean1 = df1.mean().tolist()
STotalMean2 = df2.mean().tolist()
STotalStd1 = df1.std().tolist()
STotalStd2 = df2.std().tolist()
Sn1 = len(df1)
Sn2 = len(df2)
ZStudent = []
for i in range(6):
    ZStudent.append(np.abs((STotalMean2[i]-STotalMean1[i])/(np.sqrt(((STotalStd2[i])**2/Sn2)+((STotalStd1[i])**2/Sn1)))))

print('In Student status:' , X[ZStudent.index(min(ZStudent))])
#Income  Limit         Rating         Cards            Age             Education        Balance  

Zmean = df.groupby(['Gender']).mean()
Zstd =  df.groupby(['Gender']).std()
malemean = Zmean.iloc[0].tolist()
femalemean = Zmean.iloc[1].tolist()
malestd = Zstd.iloc[0].tolist()
femalestd = Zstd.iloc[1].tolist()
ZGender = []
for i in range(6):
    ZGender.append(np.abs((malemean[i]-femalemean[i])/(np.sqrt(((malestd[i])**2/193)+((femalestd[i])**2/207)))))
print('In Gender status:' , X[ZGender.index(min(ZGender))])
