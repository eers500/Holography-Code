# -*- coding: utf-8 -*-
"""Calculate position of particles for several frames"""
#%% Import vido and set paramaters
import time
import numpy as np
import easygui
import pandas as pd
import matplotlib.pyplot as plt
import functions as f

# PATH = easygui.fileopenbox()
# PATH = 'MF1_30Hz_200us_awaysection.avi'
#PATH = '10x_laser_50Hz_10us_g1036_bl1602_500frames.avi'
PATH = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames.avi'
T0 = time.time()
VID = f.videoImport(PATH, 0)
FRAMES_MEDIAN = 20
I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)

N = 1.3226
LAMBDA = 0.642              # HeNe
MPP = 10
FS = 0.711                     # Sampling Frequency px/um
SZ = 5                       # # Step size um
NUMSTEPS = 150
THRESHOLD = 1

#%% Test positions3D
# FRAME = 20
# I = VID[:, :, FRAME]
# IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS)
# GS = f.zGradientStack(IM)  # GradientStack and RS propagator
# GS[GS < THRESHOLD] = 0
# LOCS = f.positions3D(GS)

# f.plot3D(LOCS, title='pos')

#%%  Calculate propagators, gradient stack and compute particle position ins 3D
# NUM_FRAMES = np.shape(VID)[-1]
# NUM_FRAMES = int(np.floor(np.shape(VID)[-1]/2))
NUM_FRAMES = 500
LOCS = np.empty((NUM_FRAMES, 3), dtype=object)
# INTENSITY = np.empty(NUM_FRAMES, dtype=object)

T = []
T0 = time.time()
for i in range(NUM_FRAMES):
    I = VID[:, :, i]
    # IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS).astype('float32')
    # GS = f.zGradientStack(IM).astype('float32')  
    GS, IM = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
    # GS[GS < THRESHOLD] = 0.003
    LOCS[i, 0] = f.positions3D(GS, peak_min_distance=20)
    A = LOCS[i, 0].astype('int')
    LOCS[i, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[i, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
    T.append(time.time()-T0)
    print(str(i+1)+' of '+ str(NUM_FRAMES), (time.time()-T0)/60)
print((time.time()-T0)/60)

plt.plot(np.arange(len(T)), np.array(T)/60, '.-'); plt.grid()
plt.title('Computation time per frame'); plt.xlabel('Number of frames'); plt.ylabel('Time (min)')

POSITIONS = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'])
for i in range(np.shape(LOCS)[0]):
    XYZ, I_FS, I_GS, FRAME = LOCS[i, 0], LOCS[i, 1], LOCS[i, 2], i*np.ones_like(LOCS[i, 2])
    DATA = np.concatenate((XYZ, np.expand_dims(I_FS, axis=1), np.expand_dims(I_GS, axis=1), np.expand_dims(FRAME, axis=1)), axis=1)
    POSITIONS = POSITIONS.append(pd.DataFrame(DATA, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME']))
    
# POSITIONS.to_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', header=True)
#%% Plot with f.plot3D
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot

# fig  = pyplot.figure()
# ax = Axes3D(fig)
# for i in range(NUM_FRAMES):
#     f.plot3D(LOCS[i, 0],'Cells in 3D', fig, ax); pyplot.show()
#     pyplot.pause(0.1)
    
#%% Plot with plotly.express
# import plotly.express as px
# import pandas as pd
# from plotly.offline import plot

# fig = px.scatter_3d(POSITIONS, x='X', y='Y', z='Z', color='FRAME')
# fig.update_traces(marker=dict(size=4))
# plot(fig)

#%% Plot with plotly.graph_objects
import plotly.graph_objects as go
from plotly.offline import plot

fig = go.Figure(data=[go.Scatter3d(
    x=POSITIONS['X'], 
    y=POSITIONS['Y'], 
    z=POSITIONS['Z'],
    mode='markers', 
    marker=dict(
        size=1,
        color=POSITIONS['FRAME'].values,
        colorscale='Viridis'
        ),
    hovertext=['X+Y+Z+FRAME']
    
)])
fig.show()
plot(fig)

#%% Matplotlib scatter plot
# 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot

# PKS = A.__array__()
# np.savetxt('locs.txt', PKS)
# fig = pyplot.figure()
# ax = Axes3D(fig)

# p = ax.scatter(PKS[:, 0], PKS[:, 1], PKS[:, 2], s=25, marker='o', c=PKS[:,3])
# ax.tick_params(axis='both', labelsize=10)
# ax.set_title('Cells Positions in 3D', fontsize='20')
# ax.set_xlabel('x (pixels)', fontsize='18')
# ax.set_ylabel('y (pixels)', fontsize='18')
# ax.set_zlabel('z (slices)', fontsize='18')
# fig.colorbar(p)
# pyplot.show()


