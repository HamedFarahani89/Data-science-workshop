


import numpy as np
import pylab as plt
data=np.load('sample_data.npy')
X=data[:,0]
Y=data[:,1]
error=data[:,2]
y= lambda A,B : ((A*X)+(B*(X**2)))
A,B=10,13
dA=1.5
dB=1.5
n=0
y_theory=y(A,B)
chi2=np.sum(((y_theory-Y)**2)/Y)
for i in range(500):
    y_theory1=y(A-dA,B)
    chi21=np.sum(((y_theory1-Y)**2)/Y)
    
    y_theory2=y(A,B-dB)
    chi22=np.sum(((y_theory2-Y)**2)/Y)
    
    y_theory3=y(A-dA,B-dB)
    chi23=np.sum(((y_theory3-Y)**2)/Y)
    
    INFO=np.matrix([[chi21, chi22, chi23], [A-dA, A, A-dA], [B, B-dB, B-dB]])
    MIN_loss=INFO[0].argmin()
    best_loss=INFO[:,MIN_loss]
    
    if best_loss[0,0] < chi2:
        chi2 = best_loss[0,0]
        A = best_loss[1,0]
        B = best_loss[2,0]
    else:
        n+=1
    if n > 1:
        dA = dA/10
        dB = dB/10
        n=0
F11=np.sum((X**2)/(error**2))
F12=np.sum((X**3)/(error**2))
F21=np.sum((X**3)/(error**2))
F22=np.sum((X**4)/(error**2))
F=np.array([[F11, F12],[F21, F22]])
print(A)
print(B)
print(F)
