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
