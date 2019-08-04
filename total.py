import sys



def frac(n):

	if n<0:
		print('n is negative and doesnt have factorial')
	elif n==0:
		print('1')
	else:	
		fac=1

		for i in range(1,n+1):
			fac=fac*i
			i+=1
		return fac

n=int(sys.argv[1])
print(frac(n))

##########


import sys


def prime(n):
	x=[]
	for i in range(1,n+1):
		if n%i==0:		
			x.append(i)
		
	if len(x)==2:
		print('prime')
	else:
		print('not prime')		
	

		

n=int(sys.argv[1])
print(prime(n))


##########



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


##########




import sys

f = open(sys.argv[1], "r")
g=[]

lines = f.readlines()
for line in lines:
	words = line.split()
	words = [w.replace('is', 'IS') for w in words]

	for word in words:
		 g.append(word)		
		
				
		
			
print(g)


##########
حامد فراهانی
