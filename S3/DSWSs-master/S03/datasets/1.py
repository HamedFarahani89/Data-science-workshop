#1. For a given value p0, how many times Tesla stock price (min and max average) gets more than p0. 
#dataset: tesla-stock-price.csv

#input: 
 #   p0 (float)
#output: 
 #   number of occurrence (int)


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


