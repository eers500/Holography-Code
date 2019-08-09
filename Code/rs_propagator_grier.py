# -*- coding: utf-8 -*-
#%%
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from functions import bandpassFilter, exportAVI, dataCursor
from progress.bar import Bar

T0 = time.time()
# I = mpimg.imread('131118-1.png')
I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
# I = mpimg.imread('10x_laser_50Hz_10us_g1036_bl1602-003.png')


#%%
# Median image
# IB = mpimg.imread('AVG_131118-1.png')
IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')
# IB = mpimg.imread('MED_10x_laser_50Hz_10us_g1036_bl1602-003-1.png')

IB[IB == 0] = np.mean(IB)
IN = I/IB   #divide
# IN = I - IB    #substract as in imageJ
# IN[IN < 0] = 0

N = 1.3226
LAMBDA = 0.642           #HeNe
# MPP = 2              # Magnification: 10x, 20x, 50x, etc
FS = 0.711                #Sampling Frequency px/um
NI = np.shape(IN)[0]
NJ = np.shape(IN)[1]
Z = FS/2*np.arange(1, 151)
K = 2*m.pi*N/LAMBDA      #Wavenumber

IBAND, BP = bandpassFilter(IN, 2, 30)
E = BP*np.fft.fft2(IN - 1)

#%%
# qsq = ((lambdaa/(N*n))*nx)**2 + ((lambdaa/(N*n))*ny)**2
P = np.empty_like(IB, dtype=complex)
for i in range(NI):
    for j in range(NJ):
        P[i, j] = ((LAMBDA*FS)/(max([NI, NJ])*N))**2*((i-NI/2)**2+(j-NJ/2)**2)
P = np.conj(P)
Q = np.sqrt(1-P)-1

if all(Z>0):
    Q = np.conj(Q)

R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
IZ = np.empty_like(R, dtype=float)

BAR = Bar('Processing', max=Z.shape[0])
for k in range(Z.shape[0]):
    R[:, :, k] = np.exp((-1j*K*Z[k]*Q))
    IZ[:, :, k] = 1 + np.real(np.fft.ifft2(E*R[:, :, k]))
    BAR.next()
BAR.finish()
# print(('RS', k))
IZ8 = np.uint8((IZ - np.min(IZ))*255)
IZ8 = np.uint8(IZ8/np.max(IZ8)*255)
IZ8 = np.uint8(255*(IZ8/255)**2)
# IM = 50*(IZ - 1) + 128
# IMM = IM/np.max(IM)*255
# IMM = np.uint8(IMM)

print(time.time() - T0)

#%%
# plt.imshow(IZ[:, :, 149], cmap = 'gray')

#%%
IZZ = np.abs(IZ)**2
IZZ = (IZZ-np.min(IZZ))/np.max((IZZ-np.min(IZZ)))*255
IZZZ = np.uint8(255-IZZ)
exportAVI('IZZZ.avi',IZZZ, IZZ.shape[0], IZZ.shape[1], 30)

#%%
# minI = 0.8
# maxI = 1
#
# minO = 0
# maxO = 255
#
# INO = (IN - minI)*(((maxO - minO)/(maxI - minI)) + minO)
#
# plt.imshow(INO, cmap='gray')