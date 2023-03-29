import numpy as np
K = np.zeros((18,18))
k = np.ones((6,6))
inod = 1
jnod = 3
sysdofs = [inod*3, inod*3 +1, inod*3+2, jnod*3, jnod*3+1, jnod*3+2]
K[np.ix_(sysdofs,sysdofs)] += k
print(sysdofs)
print(K)
K[3:6,3:6] += k[0:3,0:3]
print(K)