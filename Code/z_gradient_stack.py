# -*- coding: utf-8 -*-
"""Calculte gradient stack using sobel-type kernel"""
#%%
# Import libraries and resources
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from functions import rayleighSommerfeldPropagator, exportAVI

T0 = time.time()
# I = mpimg.imread('131118-1.png')
# I_MEDIAN = mpimg.imread('AVG_131118-1.png')

I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
I_MEDIAN = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')

# I = mpimg.imread('10x_laser_50Hz_10us_g1036_bl1602-003.png')
# I_MEDIAN = mpimg.imread('MED_10x_laser_50Hz_10us_g1036_bl1602-003-1.png')

N = 1.3226
LAMBDA = 0.642
MPP = 20                      # Magnification: 10x, 20x, 50x, etc
FS = 0.711                # Sampling Frequency px/um
SZ = 4                        # # Step size um
NUMSTEPS = 150

# Z = np.arange(1, 151)
IM = rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS)

# plt.imshow(np.uint8(IM[:,:,140]), cmap='gray')

#%%
# Sobel-type kernel
SZ0 = np.array(([-1, -2, -1], [-2, -4, -2], [-1, -2, -1]), dtype='float')
SZ1 = np.zeros_like(SZ0)
SZ2 = -SZ0
SZ = np.stack((SZ0, SZ1, SZ2), axis=-1)
# del SZ0, SZ1, SZ2

#%%
# Convolution IM*SZ
# IM_FFT = np.fft.fftn(np.dstack([IM[:,:,0:2], IM]))
# SZ_FFT = np.fft.fftn(SZ, IM_FFT.shape)
# PROD = IM_FFT*SZ_FFT
# CONV = np.real(np.fft.ifftn(PROD))
# CONV = (20/np.std(CONV))*(CONV - np.mean(CONV)) + 128
# CONV = np.delete(CONV, [0,1], axis=2)

#%%
# Convolution IM*SZ
# IM = IM**2        # Intensity of E field?
IMM = np.dstack((IM[:, :, 0][:, :, np.newaxis], IM, IM[:, :, -1][:, :, np.newaxis]))
GS = ndimage.convolve(IMM, SZ, mode='mirror')
GS = np.delete(GS, [0, np.shape(GS)[2]-1], axis=2)
del IMM
##
THRESHOLD = 0.3
GS[GS < THRESHOLD] = 0
GS[GS > THRESHOLD] = 255

#%%
# IS = GS[356:358, 291:293, :]
IS = GS
D = 4
NI, NJ = IS.shape[0]/D, IS.shape[1]/D
ISS = np.empty([int(NI), int(NJ), IS.shape[-1]])
for i in range(ISS.shape[0]):
    for j in range(ISS.shape[1]):
        for k in range(ISS.shape[2]):
            S = IS[D*i:D*i+D, D*i:D*i+D, :]
            ISS[i, j, k] = np.sum(S)

#%%
# For visualization
# GS = GS - np.min(GS)
# IM = IM - np.min(IM)
#
# GS = GS/np.max(GS)*255
# IM = IM/np.max(IM)*255
#
# GS = np.uint8(GS)
# IM = np.uint8(IM)

#%%
# Prepare to export AVI
# IM = 50*(IZ - 1) + 128
# IMM = np.abs(IM)**2
IMM = (IM-np.min(IM))/np.max((IM-np.min(IM)))*255
IMMM = np.uint8(IMM)

# GSS = np.abs(GS)**2
GSS = (GS-np.min(GS))/np.max((GS-np.min(GS)))*255
GSSS = np.uint8(GSS)

#%%s
# Export results as .AVI
# exportAVI('frameStack.avi', IMMM, IM.shape[0], IM.shape[1], 30)
exportAVI('gradientStack.avi', GSSS, GS.shape[0], GS.shape[1], 30)
print(time.time()-T0)
# del T0

## Plot
# G = GS[:, :, 36]
# plt.imshow(-G, cmap='gray')
# plt.title('Gradient Stack Slice #36', fontsize='20')

#%%
# Plot
# plt.imshow(GS[:,:,12], cmap='gray')
# plt.title('E.coli Raw Hologram', fontsize='20')
# plt.xlabel('x (pixels)', fontsize='18')
# plt.ylabel('y (pixels)', fontsize='18')

#%%
# 3D surace Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
#
# fig = pyplot.figure()
# ax = Axes3D(fig)
#
# X, Y = np.meshgrid(np.arange(1, 513, 1), np.arange(1, 513, 1))
# ax.plot_surface(X, Y, GS[:, :, 12])
# ax.tick_params(axis='both', labelsize=10)
# ax.set_title('Cells Positions in 3D', fontsize='20')
# ax.set_xlabel('x (pixels)', fontsize='18')
# ax.set_ylabel('y (pixels)', fontsize='18')
# ax.set_zlabel('z (slices)', fontsize='18')
# pyplot.show()

#%%
# Plot bokeh
# from bokeh.plotting import figure, show, output_file
#
#
# p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
# # p.x_range.range_padding = p.y_range.range_padding = 0
#
# # must give a vector of image data for image parameter
# im = GS[:,:,36]
# p.image(image=[im[::-1]], x=0, y=0, dw=512, dh=512, palette="Spectral11")
#
# output_file("image.html", title="image.py example")
#
# show(p)  # open a browser
#
# ##
# GG = mpimg.imread("Result of MF1_30Hz_200us_awaysection-1.png")
#
# plt.imshow(GG, cmap='gray')
# plt.title('Normalized Hologram', fontsize='20')
