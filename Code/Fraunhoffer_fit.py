import numpy as np
import scipy.special as sp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

IM = mpimg.imread('Ring_10x_laser_50Hz_10us_g1036_bl1602-12.png')

# def func(W, rho):
#     LAMBDA = 642
#     z = 10
#     return np.pi*sp.j1(np.pi*W*rho/(LAMBDA*z))
#
# X = np.arange(-IM.shape[0]/2, IM.shape[0]/2)
# Y = X
# x, y = np.meshgrid(X, Y)
# r = np.sqrt(x**2 + y**2)

#%%
r = np.linspace(1, 14, 100)
W = 1
LAMBDA = 642
z = 0.004
B = np.pi*W*r/LAMBDA
A = B/z
I = np.pi*sp.j1(A)/A
plt.plot(I)
#%%
popt, pcov = curve_fit(func, IM)
