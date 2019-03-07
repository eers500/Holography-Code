# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from functions import rayleighSommerfeldPropagator, exportAVI

T0 = time.time()
I = mpimg.imread('131118-1.png')
I_MEDIAN = mpimg.imread('AVG_131118-2.png')
Z = 0.02*np.arange(1, 151)
IM = rayleighSommerfeldPropagator(I, I_MEDIAN, Z)

#%% Sobel-type kernel
SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]), dtype='float')
SZ1 = np.zeros_like(SZ0)
SZ2 = -SZ0
SZ = np.stack((SZ0, SZ1, SZ2), axis=2)
del SZ0, SZ1, SZ2, I, I_MEDIAN, Z

#%% Convolution IM*SZ
IM_FFT = np.fft.fftn(IM)
SZ_FFT = np.fft.fftn(SZ, IM.shape)

PROD = IM_FFT*SZ_FFT
CONV = np.real(np.fft.ifftn(PROD))
CONV = (20/np.std(CONV))*(CONV - np.mean(CONV)) + 128
CONV[CONV<100] = 0
CONV = np.uint8(CONV)
#del IM_FFT, PROD

exportAVI(CONV, IM.shape[0], IM.shape[1], 24)
print(time.time()-T0)

#%%
plt.imshow(CONV[:,:,100], cmap='gray')
