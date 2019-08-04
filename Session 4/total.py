
import numpy as np
BobR=[]
ExpNo=100000
for i in range(ExpNo):
    BobR.append(np.random.choice([0,0,1], size=1))


AliceR=[]
for i in range(ExpNo):
    AliceR.append(np.random.choice([0,0,1], size=1))

print("Alice's chance probability:" ,(ExpNo-np.sum(AliceR))/ExpNo)
print("Bob's chance probability:" ,np.mean(BobR))

##########

import numpy as np
import pandas as pd


DF = pd.read_csv('UN_cleaned.csv')
DF=DF.set_index('Unnamed: 0')

DFcorr=DF.corr(method='pearson')
print(DFcorr)

##########





import numpy as np
import pylab as plt
import math


df=np.load('exercise.npy')
x_obs=df[:,0]
y_obs=df[:,1]


def model(x,a,b):
    return a*math.sin(b*x)
model2 = np.vectorize(model) 



listmin=[]
listaas=[]
listbbs=[]


for b in np.arange(-2,3,0.1):
    for a in np.arange(-2,3,0.1):
        y_pointer=model2(x_obs,a,b)

        loss=np.sum((y_pointer-y_obs)**2)
        listmin.append(loss)
        listaas.append(a)
        listbbs.append(b)

print(listaas[listmin.index(min(listmin))]), print(listbbs[listmin.index(min(listmin))])

##########
حامد فراهانی
