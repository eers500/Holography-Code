# -*- coding: utf-8 -*-
"""Calculte gradient stack using sobel-type kernel"""
#%%
# Import libraries and resources
import time
import numpy as np
import easygui
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max
from scipy import ndimage
from functions import rayleighSommerfeldPropagator, exportAVI


# I = mpimg.imread('131118-1.png')
# I_MEDIAN = mpimg.imread('AVG_131118-1.png')

PATH = easygui.fileopenbox(multiple=True)
I = mpimg.imread(PATH[0])
I_MEDIAN = mpimg.imread(PATH[1])

# I = mpimg.imread('10x_laser_50Hz_10us_g1036_bl1602-003.png')
# I_MEDIAN = mpimg.imread('MED_10x_laser_50Hz_10us_g1036_bl1602-003-1.png')

N = 1.3226
LAMBDA = 0.642
MPP = easygui.integerbox(msg='Enter magnification (10x, 20x, etc)', title='Magnification', default=10)                      # Magnification: 10x, 20x, 50x, etc
FS = 0.711                # Sampling Frequency px/um
SZ = 4                        # # Step size um
NUMSTEPS = 150

T0 = time.time()
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
# IM = IM**2        # Intensity of E field?
IMM = np.dstack((IM[:, :, 0][:, :, np.newaxis], IM, IM[:, :, -1][:, :, np.newaxis]))
GS = ndimage.convolve(IMM, SZ, mode='mirror')
GS = np.delete(GS, [0, np.shape(GS)[2]-1], axis=2)
del IMM

#%%
THRESHOLD = 0.2
# THRESHOLD = np.mean(GS)
GS[GS < THRESHOLD] = 0
# GS[GS > THRESHOLD] = 255

#%%
ZP = np.max(GS, axis=2)
PKS = peak_local_max(ZP, min_distance=3)

MAX = np.empty((len(PKS), 1))
for i in range(len(PKS)):
    M = np.where(GS[PKS[i, 0], PKS[i, 1], :] == np.max(GS[PKS[i, 0], PKS[i, 1], :]))
    MAX[i,0] = M[0][0]

PKSS = np.hstack((PKS, MAX))
print(time.time()-T0)

# plt.plot(GS[PKS[:,0], PKS[:,1], :])
# plt.imshow(ZP, cmap='gray')
# plt.scatter(PKS[:,1], PKS[:,0], marker='o', facecolors='none', s=80, edgecolors='r')
# plt.colorbar()

#%%
# Testing
ZP = np.max(GS, axis=-1)
PKS = peak_local_max(ZP, min_distance=3)

# idj = PKS[0, 0]
# idi = PKS[0, 1]
#
# A = GS[idi-3:idi+4:, idj-3:idj+4, :]
# # A = GS[PKS[0, 1]-3:PKS[0, 1]+3, PKS[0, 0]-3:PKS[0, 0]+3, :]
# B = np.sum(A, axis=(0, 1))

B = np.empty((NUMSTEPS, len(PKS)))
for ii in range(len(PKS)):
    idi = PKS[ii, 0]
    idj = PKS[ii, 1]
    A = GS[idi-3:idi+4:, idj-3:idj+4, :]
    B[:, ii] = np.sum(A, axis=(0, 1))

C = np.empty((len(PKS), 1), dtype=object)
for ii in range(len(PKS)):
    C[ii, 0] = peak_local_max(B[:, ii])
    # C.append(peak_local_max(B[:, ii]))

#%%
D = []
for ii in range(len(C)):
    if len(C[ii, 0]) != 1:
        for jj in range(len(C[ii, 0])):
            D.append([C[ii, 0][jj].item(), ii])
    else:
        D.append([C[ii, 0].item(), ii])
D = np.array(D)




#%%
# import plotly.express as px
# import pandas as pd
# from plotly.offline import plot
#
# LOCS = pd.DataFrame(data=PKS, columns=['x', 'y', 'z'])
#
# fig = px.scatter_3d(LOCS, x='x', y='y', z='z', color='z')
# fig.update_traces(marker=dict(size=3))
# plot(fig)

#%%
# 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
#
# fig = pyplot.figure()
# ax = Axes3D(fig)
#
# ax.scatter(PKS[:, 0], PKS[:, 1], PKS[:, 2], s=50, marker='o')
# ax.tick_params(axis='both', labelsize=10)
# ax.set_title('Cells Positions in 3D', fontsize='20')
# ax.set_xlabel('x (pixels)', fontsize='18')
# ax.set_ylabel('y (pixels)', fontsize='18')
# ax.set_zlabel('z (slices)', fontsize='18')
# pyplot.show()

#%%
# Prepare to export AVI
# IM = 50*(IZ - 1) + 128
# IMM = np.abs(IM)**2
# IMM = (IM-np.min(IM))/np.max((IM-np.min(IM)))*255
# IMMM = np.uint8(IMM)

# GSS = np.abs(GS)**2
# GSS = (GS-np.min(GS))/np.max((GS-np.min(GS)))*255
# GSSS = np.uint8(GSS)

#%%s
# Export results as .AVI
# exportAVI('frameStack.avi', IMMM, IM.shape[0], IM.shape[1], 30)
# exportAVI('gradientStack.avi', GSSS, GS.shape[0], GS.shape[1], 30)



