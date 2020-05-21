# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:37:50 2019

@author: erick
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from scipy import optimize
import functions as f

I = mpimg.imread('Ring_10x_laser_50Hz_10us_g1036_bl1602-12.png')
IM = 1-(I/np.max(I))
IM = zoom(IM, 1, order=2)

SIGMA = 0.5
IMM = gaussian_filter(IM, sigma=SIGMA)
plt.set_cmap('gray')
plt.close()


#%%
NI, NJ = np.shape(IM)
IHORZ = IM[int(NI/2), int(NJ/2):-1]
IHORZ_FP = np.flip(IM[int(NI/2), 0:int(NJ/2)-1])
IVERT = IM[int(NI/2):-1, int(NJ/2)]
IVERT_FP = np.flip(IM[0:int(NI/2)-1, int(NJ/2)])

IHORZ2 = IMM[int(NI/2), int(NJ/2):-1]
IHORZ2_FP = np.flip(IM[int(NI/2), 0:int(NJ/2)-1])
IVERT2 = IMM[int(NI/2):-1, int(NJ/2)]
IVERT2_FP = np.flip(IM[0:int(NI/2)-1, int(NJ/2)])

X = np.linspace(1, len(IHORZ), len(IHORZ))

#%%
# Custom curve to fit
def func(rho, wsize, zdist):
    """Fraunhofer diffraction"""
    from scipy.special import j1
    lam = 0.642     # lambda
    # z = 5
    return j1(np.pi*wsize*rho/(lam*zdist))/(wsize*rho/(lam*zdist))

#%%
# Fit
POPTHZ, PCOVHZ = optimize.curve_fit(func, X, IHORZ, bounds=(0, [2., 1500.]))
POPTVT, PVOVVT = optimize.curve_fit(func, X, IVERT, bounds=(0, [2., 1500.]))
POPTHZ_FP, PCOVHZ_FP = optimize.curve_fit(func, X, IHORZ_FP, bounds=(0, [2., 1500.]))
POPTVT_FP, PVOVVT_FP = optimize.curve_fit(func, X, IVERT_FP, bounds=(0, [2., 1500.]))

POPTHZ2, PVCOVHZ2 = optimize.curve_fit(func, X, IHORZ2, bounds=(0, [2., 1500.]))
POPTVT2, PCOVVT2 = optimize.curve_fit(func, X, IVERT2, bounds=(0, [2., 1500.]))
POPTHZ2_FP, PCOVHZ2_FP = optimize.curve_fit(func, X, IHORZ2_FP, bounds=(0, [2., 1500.]))
POPTVT2_FP, PVOVVT2_FP = optimize.curve_fit(func, X, IVERT2_FP, bounds=(0, [2., 1500.]))

#%%
FIG2, AX2 = plt.subplots(1, 2)
AX2[0].imshow(IM)
AX2[0].title.set_text('Original')
AX2[1].imshow(IMM)
AX2[1].title.set_text('Gaussian blured, sigma=%5.1f' % SIGMA)
f.dataCursor2D()

FIG, AX = plt.subplots(2, 1)
AX[0].plot(X, IHORZ, 'rs-', X, IVERT, 'bs-', IHORZ_FP, 'gs-', IVERT_FP, 'ys-')
AX[0].plot(X, func(X, *POPTHZ), 'r-', label='fit: W=%5.3f, z=%5.3f' % tuple(POPTHZ))
AX[0].plot(X, func(X, *POPTVT), 'b-', label='fit: W=%5.3f, z=%5.3f' % tuple(POPTVT))
AX[0].plot(X, func(X, *POPTHZ_FP), 'g-', label='fit: W=%5.3f, z=%5.3f' % tuple(POPTHZ_FP))
AX[0].plot(X, func(X, *POPTVT_FP), 'y-', label='fit: W=%5.3f, z=%5.3f' % tuple(POPTVT_FP))
AX[0].grid()
AX[0].legend()

AX[1].plot(X, np.abs(IHORZ-func(X, *POPTHZ)), 'rs-', label='Horizontal fit error')
AX[1].plot(X, np.abs(IVERT-func(X, *POPTVT)), 'bd-', label='Vertical fit error')
AX[1].plot(X, np.abs(IHORZ_FP-func(X, *POPTHZ_FP)), 'gs-', label='Horizontal fit error')
AX[1].plot(X, np.abs(IVERT_FP-func(X, *POPTVT_FP)), 'yd-', label='Vertical fit error')
AX[1].grid()
AX[1].legend()

FIG3, AX3 = plt.subplots(2, 1)

AX3[0].plot(X, IHORZ2, 'rs-', X, IVERT2, 'bs-')
AX3[0].plot(X, func(X, *POPTHZ2), 'r-', label='fit: W=%5.3f, z=%5.3f' % tuple(POPTHZ2))
AX3[0].plot(X, func(X, *POPTVT2), 'b-', label='fit: W=%5.3f, z=%5.3f' % tuple(POPTVT2))
AX3[0].plot(X, func(X, *POPTHZ2_FP), 'g-', label='fit: W=%5.3f, z=%5.3f' % tuple(POPTHZ2_FP))
AX3[0].plot(X, func(X, *POPTVT2_FP), 'y-', label='fit: W=%5.3f, z=%5.3f' % tuple(POPTVT2_FP))
AX3[0].grid()
AX3[0].legend()

AX3[1].plot(X, np.abs(IHORZ2-func(X, *POPTHZ2)), 'rs-', label='Horizontal fit error')
AX3[1].plot(X, np.abs(IVERT2-func(X, *POPTVT2)), 'bd-', label='Vertical fit error')
AX3[1].plot(X, np.abs(IHORZ2_FP-func(X, *POPTHZ2_FP)), 'gs-', label='Horizontal fit error')
AX3[1].plot(X, np.abs(IVERT2_FP-func(X, *POPTVT2_FP)), 'yd-', label='Vertical fit error')
AX3[1].grid()
AX3[1].legend()

#%%
# xx = np.linspace(-25, 25, 50)
# yy = xx
#
# XX, YY = np.meshgrid(xx, yy)
# T = XX + YY*1j
# THETA = np.arctan(YY, XX)
# R = np.sqrt(XX**2 + YY**2)
# THETA = np.linspace(0, 2*np.pi, 50)
