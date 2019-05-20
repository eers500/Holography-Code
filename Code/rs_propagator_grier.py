# -*- coding: utf-8 -*-
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from functions import bandpassFilter, exportAVI
from progress.bar import Bar

T0 = time.time()
#I = mpimg.imread('131118-1.png')
I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
#I = np.uint8(I/np.max(I)*255)
#%% Median image
#IB = mpimg.imread('AVG_131118-2.png')
IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')
#IB = np.uint8(IB/np.max(IB)*255)
#IB = mpimg.imread('img_gs.png')
#IB = signal.medfilt2d(I, kernel_size = 3)
#IZ = np.where(IB == 0)
IB[IB == 0] = np.average(IB)
#IN = I/IB     #divide
IN = I - IB    #substract as in imageJ
IN[IN < 0] = 0

N = 1.3226
LAMBDA = 0.642           #HeNe
FS = 1.422                #Sampling Frequency px/um
NI = np.shape(IN)[0]
NJ = np.shape(IN)[1]
Z = 0.02*np.arange(1, 151)
K = 2*m.pi*N/LAMBDA      #Wavenumber

_, BP = bandpassFilter(IN, 2, 30)
E = BP*np.fft.fft2(IN - 1)
#%%
#qsq = ((lambdaa/(N*n))*nx)**2 + ((lambdaa/(N*n))*ny)**2
P = np.empty_like(IB, dtype=complex)
for i in range(NI):
    for j in range(NJ):
        P[i, j] = ((LAMBDA*FS)/(max([NI, NJ])*N))**2*((i-NI/2)**2+(j-NJ/2)**2)
Q = np.sqrt(1-P)-1
Q = np.conj(Q)

R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
IZ = np.empty_like(R, dtype=float)

BAR = Bar('Processing', max=Z.shape[0])
for k in range(Z.shape[0]):
    R[:, :, k] = np.exp((-1j*K*Z[k]*Q))
    IZ[:, :, k] = 1 + np.real(np.fft.ifft2(E*R[:, :, k]))
    BAR.next()
BAR.finish()
#    print(('RS', k))
IZ8 = np.uint8((IZ - np.min(IZ))*255)
IZ8 = np.uint8(IZ8/np.max(IZ8)*255)
IZ8 = np.uint8(255*(IZ8/255)**2)
#IM = 50*(IZ - 1) + 128
#IMM = IM/np.max(IM)*255
#IMM = np.uint8(IMM)

print(time.time() - T0)
#%%
#plt.imshow(IMM[:,:,149], cmap = 'gray')

#%%
#IZZ = (IZ-np.min(IZ))/np.max((IZ-np.min(IZ)))*255
#IZZZ = np.uint8(IZZ)
exportAVI('IZZ.avi',IZZ, IZZ.shape[0], IZZ.shape[1], 24)
