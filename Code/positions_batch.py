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
THRESHOLD = 0.5

#%%
# FRAME = 1
# I = VID[:, :, FRAME-1]
# IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS)
# GS = f.zGradientStack(IM)  # GradientStack and RS propagator
# GS[GS < THRESHOLD] = 0
# LOCS = f.positions3D(GS)

#%%
NUM_FRAMES = np.shape(VID)[-1]
# NUM_FRAMES = 60
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
    A = A.append(pd.DataFrame(LOCS[i], columns=['x', 'y', 'z', 'frame']))
    
fig = px.scatter_3d(A, x='x', y='y', z='z', color='frame')
fig.update_traces(marker=dict(size=3))
plot(fig)