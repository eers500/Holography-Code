#%% -*- coding: utf-8 -*-
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from functions import bandpassFilter, exportAVI

T0 = time.time()
I = mpimg.imread('131118-1.png')
#I = mpimg.imread('img_gs.png')
#img = mpimg.imread('MF1_30Hz_200us_away_median.png')
#%% Median image
IB = mpimg.imread('AVG_131118-2.png')
#IB = mpimg.imread('img_gs.png')
#IB = signal.medfilt2d(I, kernel_size = 3)
#IZ = np.where(IB == 0)
IB[IB == 0] = np.average(IB)
IN = I/IB

_, BP = bandpassFilter(IN, 2, 30)

N = 1.3226
LAMBDA = 0.642           #HeNe
FS = 1.422                #Sampling Frequency px/um
NI = np.shape(IN)[0]
NJ = np.shape(IN)[1]
Z = 0.02*np.arange(1, 151)
K = 2*m.pi*N/LAMBDA      #Wavenumber

#%%
#qsq = ((lambdaa/(N*n))*nx)**2 + ((lambdaa/(N*n))*ny)**2
P = np.empty_like(IB, dtype=complex)
for i in range(NI):
    for j in range(NJ):
        P[i, j] = ((LAMBDA*FS)/(max([NI, NJ])*N))**2*((i-NI/2)**2+(j-NJ/2)**2)
Q = np.sqrt(1-P)-1
Q = np.conj(Q)

R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
IZ = R
for k in range(Z.shape[0]):
    for i in range(NI):
        for j in range(NJ):
            R[i, j, k] = np.exp((-1j*K*Z[k])*Q[i, j])
    IZ[:, :, k] = 1+np.fft.ifft2(BP*(np.fft.fft2(IN-1))*R[:, :, k])
    print(('RS' ,k, i, j))

IZ = np.real(IZ)
IM = (20/np.std(IZ))*(IZ - np.mean(IZ)) + 128

#IZZ = IZ - np.min(IZ)
#IM = (20/np.std(IZZ))*(IZZ - np.mean(IZZ)) + 128
#IM = IZZ/np.max(IZZ) *255


IM = np.uint8(IM)
#IM = IM.astype(dtype='uint8')

T = time.time() - T0

#    IZ = np.real(IZ)
#    IM = (20/np.std(IZ))*(IZ - np.mean(IZ)) + 128
#    IM = np.uint8(IM)

#%%
plt.imshow(IZ[:,:,115], cmap = 'gray')
#plt.colorbar()


#%%
#for i in range(np.shape(z)[0]):
#    plt.imshow(Im[:,:,i], cmap = 'gray')
#    plt.savefig('{}.png'.format(i+1))
#    plt.clf()
#%%
IZZ = IZ - np.min(IZ)
IZZ = np.uint8(255*IZZ/np.max(IZZ))
exportAVI('IZ.avi', IZZ, NI, NJ, 24)

print(T/60)
 