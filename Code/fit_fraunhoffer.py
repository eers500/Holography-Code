from math import *
import numpy as np
import scipy.special as sp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model

def fraunhoffer(xy_mesh, W):
    (x, y) = xy_mesh
    LAMBDA = 642
    z = 0.004
    return np.pi*sp.j1((np.pi*W*np.sqrt(x**2+y**2)/LAMBDA)/z)/((np.pi*W*np.sqrt(x**2+y**2)/LAMBDA)/z)

x = np.arange(1, 51, 1)
y = x
xy_mesh = np.meshgrid(x, y)
W = 1

IM = mpimg.imread('Ring_10x_laser_50Hz_10us_g1036_bl1602-12.png')

#%%
lmfit_model = Model(fraunhoffer)
lmfit_result = lmfit_model.fit(IM, xy_mesh=xy_mesh, W=1)

lmfit_Rsquared = 1 - lmfit_result.residual.var()/np.var(IM)

# print('Fit R-squared:', lmfit_Rsquared, '\n')
# print(lmfit_result.fit_report())

W_FIT = lmfit_result.best_values['W']

#%%
# Ring fit with W from fit
I = fraunhoffer(xy_mesh, W_FIT)

plt.subplot(1,2,1); plt.imshow(IM, cmap='gray'); plt.title('Reference Ring')
plt.subplot(1,2,2); plt.imshow(I, cmap='gray'); plt.title('Fitted Ring')


