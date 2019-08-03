

import sys
x=input("word1")
y=input("word2")
f = open(sys.argv[1], "r")

g=[]

lines = f.readlines()
for line in lines:
	words = line.split()
	words = [w.replace(x,y) for w in words]		

print(words)
