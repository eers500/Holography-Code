# -*- coding: utf-8 -*-
#%% Import libraries and resources
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
#from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from functions import rayleighSommerfeldPropagator, exportAVI

T0 = time.time()
I = mpimg.imread('131118-1.png')
I_MEDIAN = mpimg.imread('AVG_131118-1.png')

#I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
#I_MEDIAN = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')

Z = 0.2*np.arange(1, 151)
IM = rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
#plt.imshow(np.uint8(IM[:,:,140]), cmap='gray')

#%% Sobel-type kernel
SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]), dtype='float')
SZ1 = np.zeros_like(SZ0)
SZ2 = -SZ0
SZ = np.stack((SZ0, SZ1, SZ2), axis=2)
del SZ0, SZ1, SZ2, I, I_MEDIAN, Z

#%% Convolution IM*SZ
#IM_FFT = np.fft.fftn(np.dstack([IM[:,:,0:2], IM]))
#SZ_FFT = np.fft.fftn(SZ, IM_FFT.shape)
#
#PROD = IM_FFT*SZ_FFT
#CONV = np.real(np.fft.ifftn(PROD))
##CONV = (20/np.std(CONV))*(CONV - np.mean(CONV)) + 128
#CONV = np.delete(CONV, [0,1], axis=2)
#%% Convolution IM*SZ
IMM = np.dstack((IM[:,:,0][:, :, np.newaxis], IM, IM[:,:,-1][:, :, np.newaxis]))
GS = ndimage.convolve(IMM, SZ, mode='mirror')  
GS = np.delete(GS, [0, np.shape(GS)[2]-1], axis=2)
del IMM
#%%
GS[GS<0.4] = 0

## For visualization
#GS = GS - np.min(GS)
#IM = IM - np.min(IM)
#
#GS = GS/np.max(GS)*255
#IM = IM/np.max(IM)*255
##GS = 255*(GS/255)**2
##GS = np.uint8(GS)
#
#GS = 255*GS/np.max(GS)
#GS = np.uint8(GS)
#IM = np.uint8(IM)
##del IM_FFT, PROD
#
#GS = GS + 128

#%% Esport results as .AVI
#exportAVI('gradientStack.avi',GS, GS.shape[0], GS.shape[1], 24)
#exportAVI('frameStack.avi', IM, IM.shape[0], IM.shape[1], 24)
#print(time.time()-T0)
del T0

#%% Plot
#plt.imshow(IM[:,:,100], cmap='gray')

#%%
#from skimage import img_as_ubyte
#
#GSS = img_as_ubyte(GS)
#IMS = img_as_ubyte(IM/np.max(IM))