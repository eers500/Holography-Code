#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:27:33 2020

@author: erick
"""
import matplotlib as mpl
import mpld3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
mpl.rc('figure',  figsize=(10, 6))
import functions as f
import sklearn.cluster as cl
import time
from scipy.optimize import linear_sum_assignment
from hungarian_algorithm import algorithm

# DF = pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv', index_col=0)
# DF= pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', index_col=0)
DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1_500_FRAMES_MODIFIED.csv', index_col=0)

DF.index = np.arange(len(DF))
# D = DF[['X', 'Y', 'Z', 'FRAME']].values
D = DF.values
# DD = A[:, :3]

#%% K-Means for track detection
kmeans = cl.KMeans(n_clusters=D[D[:, 5] == 0].shape[0], init='k-means++').fit(DF[['X', 'Y', 'Z']])
DF['PARTICLE'] = kmeans.labels_
LINKED = DF

#%% Density Based Spatial Clustering (DBSCAN)
DBSCAN = cl.DBSCAN(eps=20, min_samples=100).fit(DF[['X', 'Y', 'Z']])
DF['PARTICLE'] = DBSCAN.labels_
LINKED = DF

#%% Mean Shift Clustering
T0 = time.time()
MEAN_SHIFT = cl.MeanShift(bandwidth=2).fit(DF[['X', 'Y', 'Z']])
DF['PARTICLE'] = MEAN_SHIFT.labels_
LINKED = DF
T = time.time()-T0
print(T)
#%% Spectral Clustering
T0 = time.time()
SPEC = cl.SpectralClustering(n_clusters=40, eigen_solver='arpack', affinity="nearest_neighbors").fit(DF[['X', 'Y', 'Z']])
DF['PARTICLE'] = SPEC.labels_
LINKED = DF
T = time.time() - T0
print(T)
#%% Mini Batch KMeans
kmeans = cl.MiniBatchKMeans(n_clusters=D[D[:, 5] == 0].shape[0], init='k-means++').fit(DF[['X', 'Y', 'Z']])
DF['PARTICLE'] = kmeans.labels_
LINKED = DF

#%% Hungrian algorithm
T0 = time.time()
FRAME_DATA = np.empty(int(DF['FRAME'].max()), dtype=object)
SIZE = np.empty(FRAME_DATA.shape[0])
for i in range(FRAME_DATA.shape[0]):
    FRAME_DATA[i] = D[D[:, 5] == i]
    SIZE[i] = FRAME_DATA[i].shape[0]

SIZES = np.empty(FRAME_DATA.shape[0])
for i in range(FRAME_DATA.shape[0]):
    if SIZE.max() != FRAME_DATA[i].shape[0]:
        DIFF = int(SIZE.max() - FRAME_DATA[i].shape[0])
        FRAME_DATA[i] = np.pad(FRAME_DATA[i], ((0, DIFF), (0, 0)), 'constant', constant_values=0)
    SIZES[i] = FRAME_DATA[i].shape[0]
        

particle_index = np.empty((int(SIZES[0]), len(FRAME_DATA)), dtype='int')
TRACKS = []
for k in range(len(FRAME_DATA)-1):
    if k == 0:
        TRACKS.append(FRAME_DATA[k])      
    print(k)
    R1 = FRAME_DATA[k]
    R2 = FRAME_DATA[k+1]
    
    COST = np.empty((int(SIZE.max()), int(SIZE.max())))
    for i in range(int(SIZE.max())):
        for j in range(int(SIZE.max())):                
            COST[i, j] = np.sqrt(sum((R1[i, :3]- R2[j, :3])**2))      
    row_ind, particle_index[:, k] = linear_sum_assignment(COST)
    TRACKS.append(R2[particle_index[:, k], :])

# print(time.time() - T0)

#%
TRACK = np.vstack(TRACKS)
TR = pd.DataFrame(TRACK, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'])
TR['PARTICLE'] = np.tile(np.arange(SIZES[0]), len(TRACKS))

print(time.time() - T0)

ZEROS = []
SHAPES = np.empty(pd.unique(TR['PARTICLE']).shape[0])
for i in range(pd.unique(TR['PARTICLE']).shape[0]):
    ZEROS.append(np.where(TR[TR['PARTICLE'] == i].X == 0)[0])
    SHAPES[i] = ZEROS[i].shape[0]

ID = np.where(SHAPES >= 50)[0]
IND  = np.in1d(TR.PARTICLE.values,  ID, invert=True)
LINKED = TR.loc[IND, :]
    
#%% Matplotlib scatter plot
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

# A = LINKED.__array__()
A = LINKED.__array__()
p = ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=1, marker='o', c=A[:, 6])
# p = ax.plot(A[:, 0], A[:, 1], A[:, 2], '.-')

ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')

fig.colorbar(p)
pyplot.show()


#%% Plotly scatter
import plotly.express as px
import pandas as pd
from plotly.offline import plot

fig = px.scatter_3d(LINKED, x='X', y='Y', z='Z', color='PARTICLE')
fig.update_traces(marker=dict(size=1))
plot(fig)

