

import sys

f = open(sys.argv[1], "r")
g=[]

lines = f.readlines()
for line in lines:
	words = line.split()
	words = [w.replace('is', 'aan') for w in words]

	for word in words:
		 g.append(word)		
		
				
		
			
print(g)
