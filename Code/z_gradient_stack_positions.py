# -*- coding: utf-8 -*-
"""Calculate RSP, GS and positions of particles in 3D"""
#%%
import time
import numpy as np
# import pyximport
from skimage.feature import peak_local_max
import functions as f

# pyximport.install()
A = f.guiImport()
#%%
PATH = A[0]
# T0 = time.time()
VID = f.videoImport(PATH, 0)
T0 = time.time()
I_MEDIAN = f.medianImage(VID, int(A[3]))

N = float(A[4])
LAMBDA = float(A[5])              # HeNe
FS = float(A[6])                     # Sampling Frequency px/um
SZ = float(A[7])                        # # Step size um
NUMSTEPS = int(A[8])
THRESHOLD = float(A[9])
MPP = float(A[10])

#%%
I = VID[:, :, int(A[2])-1]
#%%
# import threading
T0 = time.time()
IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS)
print(time.time()-T0)
#%%
GS = f.zGradientStack(IM)  # GradientStack and RS propagator
GS[GS < THRESHOLD] = 0

#%%
# From Labview code
# from functions import videoImport
# GS = videoImport('131118-1_(frame0)gradient.avi')

#%%
# Find (x,y,z) of cells
LOCS = np.zeros((1, 3))
for k in range(GS.shape[2]):
    PEAKS = peak_local_max(GS[:, :, k], indices=True)  # Check for peak radius
    ZZ = np.ones((PEAKS.shape[0], 1)) * k
    PEAKS = np.append(PEAKS, ZZ, axis=1)
    LOCS = np.append(LOCS, PEAKS, axis=0)
LOCS = np.delete(LOCS, 0, 0)

np.savetxt('locs.txt', LOCS)
print(time.time() - T0)

#%%
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2], s=25, marker='o')
ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')
pyplot.show()
