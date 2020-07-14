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
import easygui as gui
import hdbscan
from scipy.optimize import linear_sum_assignment
from hungarian_algorithm import algorithm

PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
DF = pd.read_csv(PATH, index_col=0)

# DF = pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv', index_col=0)
# DF= pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1_500_FRAMES_MODIFIED.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_2001_FRAMES_MODIFIED.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_2001_FRAMES_RS_TH019.csv', index_col=0)

DF.index = np.arange(len(DF))
# D = DF[['X', 'Y', 'Z', 'FRAME']].values
D = DF.values
# DD = A[:, :3]

#%% K-Means for track detection
kmeans = cl.KMeans(n_clusters=D[D[:, 5] == 0].shape[0], init='k-means++').fit(DF[['X', 'Y', 'Z']])
DF['PARTICLE'] = kmeans.labels_
LINKED = DF

#%% Delete small clusters
CLUSTER_SIZES = LINKED.PARTICLE.value_counts()
    
CLUSTER_NUM = CLUSTER_SIZES.index[CLUSTER_SIZES.values < 1000]

L = LINKED.drop(np.where(LINKED.PARTICLE.isin(CLUSTER_NUM).values == True)[0])
                

#%% Density Based Spatial Clustering (DBSC)
DBSCAN = cl.DBSCAN(eps=5, min_samples=8).fit(DF[['X', 'Y', 'Z']])
DF['PARTICLE'] = DBSCAN.labels_
LINKED = DF
L = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])

#%% HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=60)
cluster_labels = clusterer.fit_predict(D[:, :3])

DF['PARTICLE'] = cluster_labels
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

ID = np.where(SHAPES >= 1000)[0]
IND  = np.in1d(TR.PARTICLE.values,  ID, invert=True)
LINKED = TR.loc[IND, :]
    

#%% Smooth trajectories
# from scipy import interpolate
# from mpl_toolkits.mplot3d import Axes3D
# from scipy import ndimage

# L = LINKED[LINKED.PARTICLE == 54].values

# num_true_pts = len(L)


# num_sample_pts = len(L)
# x_sample = L[:, 0]
# y_sample = L[:, 1]
# z_sample = L[:, 2]

# jump = np.sqrt(np.diff(x_sample)**2 + np.diff(y_sample)**2 + np.diff(z_sample)**2) 
# smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
# limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
# xn, yn, zn = x_sample[:-1], y_sample[:-1], z_sample[:-1]
# xn = xn[(jump > 0) & (smooth_jump < limit)]
# yn = yn[(jump > 0) & (smooth_jump < limit)]
# zn = zn[(jump > 0) & (smooth_jump < limit)]

# # xn = xn[(jump > 0)]
# # yn = yn[(jump > 0)]
# # zn = zn[(jump > 0)]

# # plt.subplot(3, 1, 1)
# # plt.plot(x_sample)
# # plt.subplot(3, 1, 2)
# # plt.plot(y_sample)
# # plt.subplot(3, 1, 3)
# # plt.plot(z_sample)

# m = len(xn)
# smoothing_condition = (m-np.sqrt(m), m+np.sqrt(m))
# smoothing_condition = np.mean(smoothing_condition)


# tck, u = interpolate.splprep([xn,yn,zn], s=smoothing_condition, k=3)
# x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
# u_fine = np.linspace(0,1,num_true_pts)
# x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_sample, y_sample, z_sample, c=plt.cm.jet(L[:, 5]), marker='.', label='True Poinrts')
# ax.plot(x_knots, y_knots, z_knots, 'g.', markersize=20, label='Knots')
# ax.plot(x_fine, y_fine, z_fine, 'b.-', label='Smoothed Curve')

# # ax.plot(xn, yn, zn, 'b.-', label='Smoothed Curve')


# ax.legend()

# fig.show()

#%% Smoothed trajectories
LINKED = LINKED[LINKED.PARTICLE != -1]
particle_num = LINKED.PARTICLE.unique()

smoothed_curves = -np.ones((1, 4))
for pn in particle_num:
    L = LINKED[LINKED.PARTICLE == pn].values
    X = f.smooth_curve(L)
    
    if X != -1:
        smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], pn*np.ones_like(X[1])), axis=1))) 

smoothed_curves = smoothed_curves[1:, :]
smoothed_curves_df = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'PARTICLE'])


smoothed_curves_df.to_csv(PATH[:-4]+'_smooth.csv', index=False)

#%% Matplotlib scatter plot
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

A = LINKED.__array__()
# A = LINKED[LINKED.PARTICLE != -1].values

# A = L.__array__()
# L = LINKED[LINKED.PARTICLE == 3].values

# p = ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=1, marker='o', c=A[:, 6])
# p = ax.scatter(L[:, 0], L[:, 1], L[:, 2], s=1, marker='o', c=L[:, 5])
# p = ax.scatter(xn, yn, zn, s=1, marker='o')


# p = ax.scatter(smoothed_curves[:, 0], smoothed_curves[:, 1], smoothed_curves[:, 2], c=smoothed_curves[:, 3], s=1, marker='o')

# p = ax.plot(A[:, 0], A[:, 1], A[:, 2], '.-')

ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')

fig.colorbar(p)
pyplot.show()


#%% Plotly scatter plot
import plotly.express as px
import pandas as pd
from plotly.offline import plot

# CURVE= = LINKED[LINKED.PARTICLE != -1]
CURVE = pd.DataFrame(smoothed_curves, columns=['X', 'Y', 'Z', 'PARTICLE'])

# fig = px.scatter_3d(CURVE, x='X', y='Y', z='Z', color='PARTICLE')
fig = px.line_3d(CURVE, x='X', y='Y', z='Z', color='PARTICLE')
fig.update_traces(marker=dict(size=1))
plot(fig)

# fig.write_html(PATH[:-3]+'html')
# fig.write_html(PATH[:-4]+'_HA.html')
