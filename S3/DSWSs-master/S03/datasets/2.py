
#2. How many significant earthquakes occurred in a give country?
#dataset: significant-earthquakes.csv

#input: 
 #   country name (string)
#output:
 
#   number of occurrence (int)


import numpy as np
import pylab as plt
import pandas as pd
import sys
sigearth = pd.read_csv('significant-earthquakes.csv')

# this line count when satiffy with two things in data, first of all find the country that user considered and second of all, the 'Number of significant earthquakes (significant earthquakes)' should be '1'

print(len(sigearth[(sigearth['Entity']==sys.argv[1]) & (sigearth['Number of significant earthquakes (significant earthquakes)']==1)]))




