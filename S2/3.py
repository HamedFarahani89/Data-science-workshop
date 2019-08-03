

import sys

f = open(sys.argv[1], "r")
g=[]

lines = f.readlines()
for line in lines:
	words = line.split()
	for word in words:
		if word.startswith('w') and word.endswith('h'):		
			g.append(word)	
		
			
print(g)
