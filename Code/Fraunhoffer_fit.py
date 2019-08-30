import numpy as np
import scipy.special as sp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import functions as f
from scipy.ndimage import gaussian_filter
from scipy import optimize

I = mpimg.imread('Ring_10x_laser_50Hz_10us_g1036_bl1602-12.png')
IM = 1-(I/np.max(I))
IMM = gaussian_filter(IM, sigma=10)
plt.set_cmap('gray')
plt.close()
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(IM)
f.dataCursor2D()

#%%
NI, NJ = np.shape(IM)
IHORZ = IM[int(NI/2), int(NJ/2)+1:49]
IVERT = IM[int(NI/2)+1:49, int(NJ/2)]
X = np.linspace(1, 23, 23)

F = lambda W, x: np.pi*sp.j1(np.pi*W*x/(0.642*10))
FE = lambda W, x, Y: F(0.642) - Y
W0 = 2

plt.figure(1)
plt.subplot(1,2,2)
plt.plot(X, IHORZ, X, IVERT)

#%%
# Custom curve to fit
def func(rho, W):
    LAMBDA = 0.642
    z = 5
    return sp.j1(np.pi*W*rho/(LAMBDA*z))/(W*rho/(LAMBDA*z))

#%%
# Fit
popthz, pcovhz = optimize.curve_fit(func, X, IHORZ)
poptvt, pcovvt = optimize.curve_fit(func, X, IVERT)

plt.figure(1)
plt.subplot(1,2,2)
plt.plot(X, func(X, *popthz), 'r-', label='fit: W=%5.3f' % tuple(popthz))
plt.plot(X, func(X, *poptvt), 'b-', label='fit: W=%5.3f' % tuple(poptvt))
# plt.plot(X, func(X, *popt), 'r-', label='fit: W=%5.3f, Z=%5.3f' % tuple(popt))
plt.axis('tight')
plt.grid()
plt.legend()
plt.show()