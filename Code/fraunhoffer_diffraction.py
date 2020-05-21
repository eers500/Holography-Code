#%%
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from functions import imshow_sequence

x = np.arange(-10, 10, 0.1)
y = x
#%%
X, Y = np.meshgrid(x, y)

#%%
R = np.sqrt(X**2 + Y**2)
W = 1
LAMBDA = 642
Z = np.arange(0, 0.01, 0.0001)
B = np.pi*W*R/LAMBDA

U = np.empty((np.shape(R)[0], np.shape(R)[1], len(Z)))
A = U
for i in range(1, len(Z)):
    A[:, :, i] = B/Z[i]
    U[:, :, i] = np.pi*sp.j1(A[:, :, i])/A[:, :, i]
    # U = sp.j1(A)
#%%
ZZ = Z[80]
W2 = 20
H = sp.j1(np.pi*W2*R/(LAMBDA*ZZ))/(np.pi*W2*R/(LAMBDA*ZZ))
# plt.imshow(H, cmap='gray')
# plt.colorbar()

imshow_sequence(U, 0.1, 1)
