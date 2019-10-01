# -*- coding: utf-8 -*-
"""Calculate position of particles for several frames"""
#%%
import time
import numpy as np
import functions as f

#%%
PATH = 'MF1_30Hz_200us_awaysection.avi'
#PATH = '10x_laser_50Hz_10us_g1036_bl1602_500frames.avi'
T0 = time.time()
VID = f.videoImport(PATH, 0)
FRAMES_MEDIAN = 20
I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)

N = 1.3226
LAMBDA = 0.642              # HeNe
MPP = 20
FS = 0.711                     # Sampling Frequency px/um
SZ = 10                       # # Step size um
NUMSTEPS = 150
THRESHOLD = 0.2

#%%
# FRAME = 1
# I = VID[:, :, FRAME-1]
# IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS)
# GS = f.zGradientStack(IM)  # GradientStack and RS propagator
# GS[GS < THRESHOLD] = 0
# LOCS = f.positions3D(GS)

#%%
# NUM_FRAMES = np.shape(VID)[-1]
# NUM_FRAMES = int(np.floor(np.shape(VID)[-1]/2))
NUM_FRAMES = 50
LOCS = np.empty(NUM_FRAMES, dtype=object)

T0 = time.time()
for i in range(NUM_FRAMES):
    print(i, (time.time()-T0)/60)
    I = VID[:, :, i]
    IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS)
    GS = f.zGradientStack(IM)
    GS[GS < THRESHOLD] = 0
    LOCS[i] = f.positions3D(GS, i)
print((time.time()-T0)/60)

#%%
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
#
# fig  = pyplot.figure()
# ax = Axes3D
# for i in range(NUM_FRAMES):
#     f.plot3D(LOCS[i],'Cells in 3D', fig, ax)

#%%
import plotly.express as px
import pandas as pd
from plotly.offline import plot

A = pd.DataFrame(columns=['x', 'y', 'z', 'frame'])
for i in range(np.shape(LOCS)[0]):
    B = LOCS[i]
    C = i*np.ones((len(B),1))
    D = np.hstack((B, C))
    A = A.append(pd.DataFrame(D, columns=['x', 'y', 'z', 'frame']))


fig = px.scatter_3d(A, x='x', y='y', z='z', color='frame')
fig.update_traces(marker=dict(size=4))
plot(fig)

#%%
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

PKS = A.__array__()
np.savetxt('locs.txt', PKS)
fig = pyplot.figure()
ax = Axes3D(fig)

p = ax.scatter(PKS[:, 0], PKS[:, 1], PKS[:, 2], s=25, marker='o', c=PKS[:,3])
ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')
fig.colorbar(p)
pyplot.show()

#%%
# import numpy as np
# import matplotlib.pyplot as plt
#
# A = np.zeros((100, 100))
# B = 255*np.ones((10, 10))
#
# A[0:10, 0:10] = B
#
# NI, NJ = np.shape(A)
#
# N = 20
# C = np.zeros((NI, NJ, N+1))
# C[:, :, 0] = A
# for ii in range(N):
#     x = np.random.randint(0, 9)*10
#     y = np.random.randint(0, 9)*10
#     C[:, :, ii+1] = np.roll(A, (x, y), axis=(0, 1))
#     # plt.imshow(C[:, :, ii+1])
#     # plt.pause(0.1)
#
# C_FT = np.fft.fftshift(np.fft.fft2(C, axes=(0, 1)))
# for ii in range(N+1):
#     plt.imshow(C[:, :, ii])
#     # plt.imshow(np.real(C_FT[:, :, ii]))
#     plt.pause(0.1)
#
# plt.close()

