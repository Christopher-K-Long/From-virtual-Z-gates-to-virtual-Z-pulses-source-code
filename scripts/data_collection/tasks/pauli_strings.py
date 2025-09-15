import numpy as np

I = np.identity(2)
X = np.array([[0,  1],
              [1,  0]])
Y = np.array([[0, -1j],
              [1j, 0]])
Z = np.array([[1,  0],
              [0, -1]])

IZ = np.kron(I, Z)
ZI = np.kron(Z, I)
IX = np.kron(I, X)
XI = np.kron(X, I)

XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)