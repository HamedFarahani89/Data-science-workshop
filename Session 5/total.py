


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
##########



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
##########






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

##########
حامد فراهانی
