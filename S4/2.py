


import numpy as np
import pandas as pd


DF = pd.read_csv('UN_cleaned.csv')
DF=DF.set_index('Unnamed: 0')

DFcorr=DF.corr(method='pearson')
print(DFcorr)
