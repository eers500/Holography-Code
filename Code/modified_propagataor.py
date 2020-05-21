#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:20:23 2020

@author: erick
"""
#%%
import os
import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import easygui
from matplotlib.widgets import Slider
from skimage.feature import peak_local_max
from functions import bandpassFilter, rayleighSommerfeldPropagator, imshow_sequence, histeq, imshow_slider, medianImage

T0 = time.time()
FILES = easygui.fileopenbox(multiple=True)

PATH = [[],[]]
if FILES[0].find('MED') != -1:
    PATH[0] = FILES[1]
    PATH[1] = FILES[0]
else:
    PATH = FILES

# PATH_I = 'FRAMES/10um_10x_ECOLI_HCB1_100hz_40us.png'
# I = mpimg.imread('131118-1.png')
# I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
# I = mpimg.imread('10x_laser_50Hz_10us_g1036_bl1602-003.png')
# I = mpimg.imread('23-09-19_ECOLI_HCB1_100Hz_50us_10x_3-frame(0).png')

I = mpimg.imread(PATH[0])

#%%
# Median image
# PATH_IB = easygui.fileopenbox()
# PATH_IB = 'FRAMES/MED_0um_10x_ECOLI_HCB1_100hz_40us-1.png'

# IB = mpimg.imread('AVG_131118-1.png')
# IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')
# IB = mpimg.imread('MED_10x_laser_50Hz_10us_g1036_bl1602-003-1.png')
# IB = mpimg.imread('MED_23-09-19_ECOLI_HCB1_100Hz_50us_10x_3-1.png')
IB = mpimg.imread(PATH[1])


IB[IB == 0] = np.mean(IB)
IN = I/IB   #divide
# PATH = '/home/erick/Documents/PhD/23_10_19/300_L_10x_100Hz_45us.tif'
# IN = mpimg.imread(PATH)


N = 1.3226
LAMBDA = 0.642               # Diode
#MPP = 20                      # Magnification: 10x, 20x, 50x, etc
FS = 0.711                     # Sampling Frequency px/um
NI = np.shape(IN)[0]
NJ = np.shape(IN)[1]
SZ = 5                       # Step size in um
# Z = (FS*(51/31))*np.arange(0, 150)       # Number of steps
Z = SZ*np.arange(0, 150)
# ZZ = np.linspace(0, SZ*149, 150)
# Z = FS*ZZ
K = 2*m.pi*N/LAMBDA            # Wavenumber

_, BP = bandpassFilter(IN, 2, 30)
E = np.fft.fftshift(BP)*np.fft.fftshift(np.fft.fft2(IN - 1))
# E = np.fft.fftshift(np.fft.fft2(IN - 1))

#%%
# Modified Propagator
Q = np.empty_like(IN, dtype='complex64')
for i in range(NI):
    for j in range(NJ):
        # Q[i, j] = 1j*(K**2 - ((i-N/2)**2 + (j-N/2)**2))**(1/2) * np.exp((K**2 - ((i-N/2)**2 + (j-N/2)**2))**(1/2))
        Q[i, j] = ((LAMBDA*FS)/(max([NI, NJ])*N))**2*((i-NI/2)**2+(j-NJ/2)**2)

Q = np.sqrt(1 - Q) - 1      

if all(Z > 0):
    Q = np.conj(Q)
 
R = np.empty([NI, NJ, Z.shape[0]], dtype='complex64')
GS = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

R1 = np.empty([NI, NJ, Z.shape[0]], dtype='complex64')
IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

# T0 = time.time()
for k in range(Z.shape[0]):
    R[:, :, k] = 1j*K*Q*np.exp((1j*K*Z[k]*Q), dtype='complex64')      # Modified 
    GS[:, :, k] = np.abs(1+np.fft.ifft2(np.fft.ifftshift(E*R[:, :, k])))
    
    R1[:, :, k] = np.exp((-1j*K*Z[k]*Q), dtype='complex64')           # Rayleigh Sommerfeld propagator
    IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E*R1[:, :, k])))

# Histogram equalization gradient stack
GS, _ = histeq(GS)
imshow_sequence(GS, 0.1, 1)

# Histogram equalization image stack
IZ, _ = histeq(IZ)
imshow_sequence(IZ, 0.1, 1)

#%% Testing 
# IZ_NORM = np.empty_like(IZ)
# IZ_MED = medianImage(IZ, 150)
# for i in range(IZ.shape[2]):
#     IZ_NORM[:, :, i] = IZ[:, :, i] / IZ_MED

# imshow_sequence(IZ_NORM, 0.1, 1)

#%%

# Set thresold for GS for particle tracking
# THRESHOLD = 1
# GS[GS < THRESHOLD] = 0

#%% Testing
# GSS, _ = histeq(GS)
# GSSS = GSS
# TH = 254.9
# GSSS[GSSS < TH] = 0
# GSSS = 255*((GSSS - TH)/np.max(GSSS - TH))
# GSSS[GSSS < 250] = 0
# imshow_sequence(GS, 0.1, 1)

# GRAD = np.gradient(IZ, axis=-1)
# GRADS, _ = histeq(GRAD)

# print(time.time() - T0)

# GS = GSSS

#%% Compute coordinates of particles using GS
# Find Z coordinates for each (x, y) found by peak_local_max in Z-projection of GS
ZP = np.max(GS, axis=-1)
PKS = peak_local_max(ZP, min_distance=30)

plt.imshow(ZP, cmap='gray')
plt.scatter(PKS[:,1], PKS[:,0], marker='o', facecolors='none', s=80, edgecolors='r')
plt.show()

D1 = 8
D2 = 8
Z_SUM_XY = np.empty((GS.shape[2], len(PKS)))
for ii in range(len(PKS)):
    idi = PKS[ii, 0]
    idj = PKS[ii, 1]
    A = GS[idi-D1:idi+D2:, idj-D1:idj+D2, :]                # How to treat borders?
    Z_SUM_XY[:, ii] = np.sum(A, axis=(0, 1))

Z_SUM_XY_MAXS_FOLDED = np.empty((len(PKS), 1), dtype=object)
for ii in range(len(PKS)):
    Z_SUM_XY_MAXS_FOLDED[ii, 0] = peak_local_max(Z_SUM_XY[:, ii], num_peaks=1)

Z_SUM_XY_MAXS = []
for ii in range(len(Z_SUM_XY_MAXS_FOLDED)):
    if len(Z_SUM_XY_MAXS_FOLDED[ii, 0]) != 1:
        for jj in range(len(Z_SUM_XY_MAXS_FOLDED[ii, 0])):
            Z_SUM_XY_MAXS.append([Z_SUM_XY_MAXS_FOLDED[ii, 0][jj].item(), ii])
    else:
        Z_SUM_XY_MAXS.append([Z_SUM_XY_MAXS_FOLDED[ii, 0].item(), ii])

Z_SUM_XY_MAXS = np.array(Z_SUM_XY_MAXS)
XYZ_POSITIONS = np.insert(PKS, 2, Z_SUM_XY_MAXS[:, 0], axis=-1)

print(time.time()-T0)

#%%
# plot with Matplotlib.pyplot
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(XYZ_POSITIONS[:, 1], XYZ_POSITIONS[:, 0], XYZ_POSITIONS[:, 2], s=50, marker='o')
ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')
pyplot.show()

#%%
# Plot with Plotly (shown as html)
import plotly.express as px
import pandas as pd
from plotly.offline import plot

# LOCS = pd.DataFrame(data=PKSS, columns=['x', 'y', 'z'])
LOCS = pd.DataFrame(data=XYZ_POSITIONS, columns=['y', 'x', 'z', 'I'])


fig = px.scatter_3d(LOCS, x='x', y='y', z='z', color='z')
fig.update_traces(marker=dict(size=3))
plot(fig)