


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
