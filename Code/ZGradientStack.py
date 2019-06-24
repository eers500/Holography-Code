# -*- coding: utf-8 -*-
## Import libraries and resources
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
#from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from functions import rayleighSommerfeldPropagator, exportAVI

T0 = time.time()
# I = mpimg.imread('131118-1.png')
# I_MEDIAN = mpimg.imread('AVG_131118-1.png')

I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
I_MEDIAN = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')

Z = 0.2*np.arange(1, 151)
IM = rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
# plt.imshow(np.uint8(IM[:,:,140]), cmap='gray')

## Sobel-type kernel
SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]), dtype='float')
SZ1 = np.zeros_like(SZ0)
SZ2 = -SZ0
SZ = np.stack((SZ0, SZ1, SZ2), axis=2)
del SZ0, SZ1, SZ2, Z

## Convolution IM*SZ
# IM_FFT = np.fft.fftn(np.dstack([IM[:,:,0:2], IM]))
# SZ_FFT = np.fft.fftn(SZ, IM_FFT.shape)
# PROD = IM_FFT*SZ_FFT
# CONV = np.real(np.fft.ifftn(PROD))
# CONV = (20/np.std(CONV))*(CONV - np.mean(CONV)) + 128
# CONV = np.delete(CONV, [0,1], axis=2)

## Convolution IM*SZ
IMM = np.dstack((IM[:,:,0][:, :, np.newaxis], IM, IM[:,:,-1][:, :, np.newaxis]))
GS = ndimage.convolve(IMM, SZ, mode='mirror')  
GS = np.delete(GS, [0, np.shape(GS)[2]-1], axis=2)
del IMM
##
THRESHOLD = 0.2
GS[GS<THRESHOLD] = 0

## For visualization
# GS = GS - np.min(GS)
# IM = IM - np.min(IM)

# GS = GS/np.max(GS)*255
# IM = IM/np.max(IM)*255
# GS = 255*(GS/255)**2
# GS = np.uint8(GS)

# GS = 255*GS/np.max(GS)
# GS = np.uint8(GS)
# IM = np.uint8(IM)
# del IM_FFT, PROD

# GS = GS + 128

## Esport results as .AVI
# exportAVI('gradientStack.avi',GS, GS.shape[0], GS.shape[1], 24)
# exportAVI('frameStack.avi', IM, IM.shape[0], IM.shape[1], 24)
# print(time.time()-T0)
del T0

## Plot
plt.imshow(GS[:,:,12], cmap='gray')
plt.title('E.coli Raw Hologram', fontsize='20')
plt.xlabel('x (pixels)', fontsize='18')
plt.ylabel('y (pixels)', fontsize='18')

## 3D surace Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

X, Y = np.meshgrid(np.arange(1, 513, 1), np.arange(1, 513, 1))
ax.plot_surface(X, Y, GS[:, :, 12])
ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')
pyplot.show()

## Surface plot bokeh
from bokeh.plotting import figure, show, output_file


p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
# p.x_range.range_padding = p.y_range.range_padding = 0

# must give a vector of image data for image parameter
im = GS[:,:,36]
p.image(image=[im[::-1]], x=0, y=0, dw=512, dh=512, palette="Spectral11")

output_file("image.html", title="image.py example")

show(p)  # open a browser
