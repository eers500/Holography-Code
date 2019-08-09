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

#%%
# Hologram dimensions
SZ = np.shape(IN)
NX = SZ[0]
NY = SZ[1]

#%%
# Parameters
N = 1.3226
LAMBDA = 0.642/N
MPP = 0.711/20
K = 2*m.pi*MPP/LAMBDA
# Z = np.arange(1, 151)/MPP*3
Z = np.linspace(1, 150, 150)
NZ = len(Z)

#%%
# Phase factor for Rayleigh-Sommerfeld propagator in Fourier space
QX = np.arange(NX) - NX/2
QX *= LAMBDA / (NX*MPP)
QSQ = QX**2

QY = np.arange(NY) - NY/2
QY *= LAMBDA / (NY*MPP)

A = np.ones((NY,))
B = np. ones((NX,))
QSQ = np.matmul(np.asmatrix(QSQ).T, np.asmatrix(A)) + np.matmul(np.asmatrix(B).T, np.asmatrix(QY**2))
QSQ = np.asarray(QSQ)

QFACTOR = K*np.sqrt(1-QSQ).astype(complex)
IKAPPA = 1j * np.real(QFACTOR)
GAMMA = np.imag(QFACTOR)

IBAND, BP = bandpassFilter(IN, 2, 30)
# BP = np.ones_like(BP)
E = np.fft.fftshift(BP)*np.fft.fftshift(np.fft.fft2(IN - 1))
RES = np.empty([NX, NY, Z.shape[0]], dtype=complex)
THISE = RES

for j in range(NZ-1):
    Hqz = np.exp(IKAPPA * Z[j] - GAMMA * abs(Z[j]))
    THISE = E * Hqz
    THISE = np.fft.ifft2(np.fft.ifftshift(THISE))
    RES[:, :, j] = THISE

#%%
IZ = np.real(RES)
IM = 50*(IZ - 1) + 128
print(time.time() - T0)

#%%
# IZZ = np.abs(IZ)**2
IZZ = (IZ-np.min(IZ))/np.max((IZ-np.min(IZ)))*255
# IZZ = (IM-np.min(IM))/np.max((IM-np.min(IM)))*255
IZZZ = np.uint8(IZZ)
exportAVI('IZZZ.avi',255-IZZZ, IZZ.shape[0], IZZ.shape[1], 30)