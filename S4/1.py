


import numpy as np
BobR=[]
ExpNo=100000
for i in range(ExpNo):
    BobR.append(np.random.choice([0,0,1], size=1))


AliceR=[]
for i in range(ExpNo):
    AliceR.append(np.random.choice([0,0,1], size=1))

print("Alice's chance probability:" ,(ExpNo-np.sum(AliceR))/ExpNo)
print("Bob's chance probability:" ,np.mean(BobR))
