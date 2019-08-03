
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






