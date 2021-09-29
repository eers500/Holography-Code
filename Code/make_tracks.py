#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:27:33 2020

@author: erick
"""
#%% Import
import os
import matplotlib as mpl
# import mpld3
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
# from hungarian_algorithm import algorithm

PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
# DF = pd.read_csv(PATH, index_col=1)
DF = pd.read_csv(PATH)
DF = DF[['X','Y','Z','I_FS','I_GS','FRAME']]
# DF = DF.head(1000000)
# DF = pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv', index_col=0)
# DF= pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1_500_FRAMES_MODIFIED.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_2001_FRAMES_MODIFIED.csv', index_col=0)
# DF = pd.read_csv('/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_2001_FRAMES_RS_TH019.csv', index_col=0)

DF.index = np.arange(len(DF))
# D = DF[['X', 'Y', 'Z', 'FRAME']].values
D = DF.values
# DD = A[:, :3]



#%% Run track detection
# KMeans, DBSCAN or HA
# method = 'DBSCAN'
method = gui.choicebox(msg='Choose a tracking Algorithm', title='Choose', choices=['KMeans', 'DBSCAN', 'HA', 'HDBSCAN'])


if method == 'KMeans':

    #% K-Means for track detection
    T0_kmeans = time.time()
    kmeans = cl.KMeans(n_clusters=D[D[:, 5] == 0].shape[0], init='k-means++').fit(DF[['X', 'Y', 'Z']])
    DF['PARTICLE'] = kmeans.labels_
    LINKED = DF
    T_kmeans = time.time() -  T0_kmeans
    print('T_kmeans', T_kmeans)

elif method == 'DBSCAN':
    
    cores = os.cpu_count()
    #% Density Based Spatial Clustering (DBSC)
    eps, min_samples = gui.multenterbox(msg='DBSCAN parameters',
                            title='DBSCAN parameters',
                            fields=['EPSILON (e.g. 5):',
                                    'MIN SAMPLES (e.g. 8):']) 
    # time.sleep(10)
    T0_DBSCAN = time.time()
    DBSCAN = cl.DBSCAN(eps=int(eps), min_samples=int(min_samples), n_jobs=cores).fit(DF[['X', 'Y', 'Z']])
    DF['PARTICLE'] = DBSCAN.labels_
    LINKED = DF
    L = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])
    T_DBSCAN = time.time() - T0_DBSCAN
    print('T_DBSCAN', T_DBSCAN)
    
elif method == 'HDBSCAN':
    
    cores = os.cpu_count()
    #% Density Based Spatial Clustering (DBSC)
    # eps, min_samples = gui.multenterbox(msg='DBSCAN parameters',
    #                         title='DBSCAN parameters',
    #                         fields=['EPSILON (e.g. 5):',
                                    # 'MIN SAMPLES (e.g. 8):']) 
    # time.sleep(10)
    T0_HDBSCAN = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, cluster_selection_epsilon=5, core_dist_n_jobs=cores,  algorithm='boruvka_kdtree')
    DF['PARTICLE'] = clusterer.fit_predict(DF[['X','Y','Z']])
    LINKED = DF.copy()
    L = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])
    T_HDBSCAN = time.time() - T0_HDBSCAN
    print('T_HDBSCAN', T_HDBSCAN)
    

elif method == 'HA':

    #% Hungrian algorithm
    T0_HA = time.time()
    FRAME_DATA = np.empty(int(DF['FRAME'].max()), dtype=object)
    SIZE = np.empty(FRAME_DATA.shape[0])
    for i in range(FRAME_DATA.shape[0]):
        FRAME_DATA[i] = D[D[:, 5] == i]
        SIZE[i] = FRAME_DATA[i].shape[0]
        
    # To make a square cost matrix
    # SIZES = np.empty(FRAME_DATA.shape[0])
    # for i in range(FRAME_DATA.shape[0]):
    #     if SIZE.max() != FRAME_DATA[i].shape[0]:
    #         DIFF = int(SIZE.max() - FRAME_DATA[i].shape[0])
    #         FRAME_DATA[i] = np.pad(FRAME_DATA[i], ((0, DIFF), (0, 0)), 'constant', constant_values=0)
    #     SIZES[i] = FRAME_DATA[i].shape[0]
            
    
    # particle_index = np.empty((int(SIZES[0]), len(FRAME_DATA)), dtype='int')
    TRACKS = [FRAME_DATA[0]]
    sizes = np.empty(len(FRAME_DATA), dtype='int')
    sizes[0] = len(FRAME_DATA[0])
    for k in range(len(FRAME_DATA)-1):  
        print(k)
        
        # if k==0:    
        #     R1 = FRAME_DATA[k]
        #     R2 = FRAME_DATA[k+1]
        # else:
        #     R1 = TRACKS[k]
        #     R2 = FRAME_DATA[k+1]
        
        R1 = FRAME_DATA[k]
        R2 = FRAME_DATA[k+1]
        
        # COST = np.empty((int(SIZE.max()), int(SIZE.max())))
        # COST = np.empty((len(R1), len(R2)))
        # for i in range(int(SIZE.max())):
        #     for j in range(int(SIZE.max())):     
      #%
        x2, x1 = np.meshgrid(R2[:, 0], R1[:, 0])
        y2, y1 = np.meshgrid(R2[:, 1], R1[:, 1])
        z2, z1 = np.meshgrid(R2[:, 2], R1[:, 2])        
        COST = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                     
        # for i in range(len(R1)):
        #     for j in range(len(R2)):
                
        #         COST[i, j] = np.sqrt(sum((R1[i, :3]- R2[j, :3])**2))      
        # row_ind, particle_index[:, k] = linear_sum_assignment(COST)
        row_ind, particle_index = linear_sum_assignment(COST)
        sizes[k+1] = len(particle_index)
        # print(len(particle_index))
        # TRACKS.append(R2[particle_index[:, k], :])
        TRACKS.append(R2[particle_index, :])
    
    # print(time.time() - T0)
    
    #%
    TRACK = np.vstack(TRACKS)
    TR = pd.DataFrame(TRACK, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'])
    # TR['PARTICLE'] = np.tile(np.arange(SIZES[0]), len(TRACKS))
    
    particle_column = []
    for i in range(len(sizes)):
        particle_column.append(np.arange(sizes[i]))    
    
    particle_column = np.concatenate(particle_column, axis=0)
    TR['PARTICLE'] = particle_column
    LINKED = TR
    T_HA = time.time() - T0_HA
    print(T_HA)

LINKED = LINKED[LINKED.PARTICLE != -1]
# ID = np.where(SHAPES >= 1000)[0]
# IND  = np.in1d(TR.PARTICLE.values,  ID, invert=True)
# LINKED = TR.loc[IND, :]

#%% Smooth trajectories
# LINKED = LINKED[LINKED.PARTICLE != -1]

# For Archea data that looks messy
# value_counts = LINKED.PARTICLE.value_counts()
# values = value_counts[value_counts.values < 1000]
# idx = LINKED.PARTICLE.isin(values.index)
# LINKED = LINKED.iloc[idx.values]

spline_degree = 3  # 3 for cubic spline
particle_num = np.sort(LINKED.PARTICLE.unique())
T0_smooth = time.time()
smoothed_curves = -np.ones((1, 4))
for pn in particle_num:
    # Do not use this
    # L = LINKED[LINKED.PARTICLE == pn].values
    # X = f.smooth_curve(L, spline_degree=spline_degree, lim=20, sc=3000)
    
    L = LINKED[LINKED.PARTICLE == pn]
    if len(L) < 100:
        continue
    X = f.csaps_smoothing(L, smoothing_condition=0.99999, filter_data=True)
    
    if X != -1:
        smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], pn*np.ones_like(X[1])), axis=1))) 

smoothed_curves = smoothed_curves[1:, :]
smoothed_curves_df = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'PARTICLE'])
T_smooth = time.time() - T0_smooth

# smoothed_curves_df.to_csv(PATH[:-4]+'_DBSCAN_smooth_200.csv', index=False)

#%% Ramer-Douglas-Peucker Algorithm to detect turns
# from rdp import rdp
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# # L = LINKED[LINKED.PARTICLE == 6]
# L = smoothed_curves_df[smoothed_curves_df.PARTICLE == 9]
# xx, yy, zz = L.X.values, L.Y.values, L.Z.values
# xx = xx.reshape(len(xx), 1)
# yy = yy.reshape(len(yy), 1)
# zz = zz.reshape(len(zz), 1)

# t = np.linspace(0, 1, len(xx))
# t = t.reshape(len(t), 1)

# stack =  np.hstack((xx, yy, zz))

# eps = 0.03
# eps3d = 2
# xyz = rdp(stack, epsilon=eps3d)
# x = rdp(np.hstack((t, xx)), epsilon=eps)
# y = rdp(np.hstack((t, yy)), epsilon=eps)
# z = rdp(np.hstack((t, zz)), epsilon=eps)

# fig = plt.figure(figsize=(7, 4.5))
# ax0 = plt.subplot2grid((6,6), (0, 0), 2, 3)
# ax0.plot(t, xx, 'b-')
# ax0.plot(x[:, 0], x[:, 1], 'r.')
# ax0.set_ylabel('x')
# ax0.set_title('eps = '+np.str(eps))

# ax1 = plt.subplot2grid((6, 6), (2, 0), 2, 3)
# ax1.plot(t, yy, 'b-')
# ax1.plot(y[:, 0], y[:, 1], 'r.')
# ax1.set_ylabel('y')

# ax2 = plt.subplot2grid((6, 6), (4, 0), 2, 3)
# ax2.plot(t, zz, 'b-')
# ax2.plot(z[:, 0], z[:, 1], 'r.')
# ax2.set_ylabel('z')

# ax3 = plt.subplot2grid((6, 6), (0, 3), 3, 4, projection='3d')
# ax3.set_facecolor('none')
# ax3.plot(L.X, L.Y, L.Z, 'b-', label='original data')
# ax3.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'r.', label='Turns')
# ax3.set_title('Smoothed track')

# ax4 = plt.subplot2grid((6, 6), (3, 3), 3, 4, projection='3d')
# ax4.set_facecolor('none')
# ax4.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', label='RDP reconstructed eps = '+np.str(eps3d))
# ax4.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'r.', label='Turns')
# ax4.set_title('RDP reconstruction')
# ax4.legend(loc='best')

# # Use derivatives of r (distance to origin) to sense changes in direction
# from scipy.signal import find_peaks
# r = np.sqrt(L.X.values**2 + L.Y.values**2 + L.Z.values**2)
# fig = plt.figure()
# ax = fig.add_subplot(311)
# ax.plot(r)

# diff = np.diff(r)
# diff2 = np.diff(diff)
# peaks, _ = find_peaks(diff, height=0.04)
# peaks2, _ = find_peaks(diff2, height=np.min(diff2)-1)   # to get all local maxima

# axx = fig.add_subplot(312)
# axx.plot(np.arange(len(diff)), diff)
# axx.plot(peaks, diff[peaks], '.')

# axx = fig.add_subplot(313)
# axx.plot(np.arange(len(diff2)), diff2)
# axx.plot(peaks2, diff2[peaks2], '.')

# fig = plt.figure()
# axxx = fig.add_subplot(121, projection='3d')
# axxx.plot(L.X, L.Y, L.Z, 'b-', label='original data')
# axxx.plot(L.X.values[peaks], L.Y.values[peaks], L.Z.values[peaks], 'r.')

# axxx = fig.add_subplot(122, projection='3d')
# axxx.plot(L.X, L.Y, L.Z, 'b-', label='original data')
# axxx.plot(L.X.values[peaks2], L.Y.values[peaks2], L.Z.values[peaks2], 'r.')

# plt.show()

#%% Matplotlib scatter plot to compare detected points with smoothed curve
# 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
# #%matplotlib qt

# p_number = 0
# CURVE_1 = LINKED[LINKED.PARTICLE == p_number]
# CURVE_2 = smoothed_curves_df[smoothed_curves_df.PARTICLE == p_number]
# # CURVE_2 = smoothed_curves_df



# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(CURVE_1.X, CURVE_1.Y, CURVE_1.Z, 'r.', label='Detected Positions', c=np.arange(len(CURVE_1.X)))
# ax.scatter(CURVE_2.X, CURVE_2.Y, CURVE_2.Z, label='Detected Positions', c=np.arange(len(CURVE_2.X)), s=0.5, marker='.')
# # ax.plot(CURVE_1.X, CURVE_2.Y, CURVE_2.Z, 'r-', label='Smoothed Curve')
# ax.plot(CURVE_2.X, CURVE_2.Y, CURVE_2.Z, 'r-', label='Smoothed Curve')
# # ax.plot(CURVE_2.X[CURVE_2.X>350], CURVE_2.Y[CURVE_2.X>350], CURVE_2.Z[CURVE_2.X>350], 'r-', label='Smoothed Curve')
# ax.set_xlabel('Y')
# ax.set_ylabel('X')
# ax.set_zlabel('Z')
# ax.set_title('Holography')
# pyplot.show()

#%% Matplotlib scatter plot
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

# PKS = A.__array__()
# np.savetxt('locs.txt', PKS)
fig = pyplot.figure()
ax = Axes3D(fig)

X = LINKED.Y
Y = LINKED.X
Z = LINKED.Z
T = LINKED.FRAME
P = LINKED.PARTICLE


ax.scatter(X, Y, Z, s=25, marker='o', c=P)
ax.plot(smoothed_curves_df.Y, smoothed_curves_df.X, smoothed_curves_df.Z, 'r-')
ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')
# fig.colorbar(p)
pyplot.show()
    

#%%
# from scipy import interpolate
# from mpl_toolkits.mplot3d import Axes3D
# from scipy import ndimage

# vals = CURVE_1.values
# x, y, z, t = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 5]
# x0, y0, z0, t0 = x[0], y[0], z[0], t[0]

# for i in range(len(x)-1):
#     dr = np.sqrt((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2 + (z[i] - z[i+1])**2)
    


#%% Plotly scatter plot
import plotly.express as px
import pandas as pd
import easygui as gui
from plotly.offline import plot

# PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', filetypes='.csv')
# smoothed_curves_df = pd.read_csv(PATH)

# For Archea data that looks messy
# value_counts = LINKED.PARTICLE.value_counts()
# values = value_counts[value_counts.values < 2000]
# idx = LINKED.PARTICLE.isin(values.index)
# CURVE = LINKED.iloc[idx.values]

# CURVE = DF
# CURVE = LINKED[LINKED.PARTICLE == 4]
# CURVE = LINKED[(LINKED.PARTICLE != -1)]
# CURVE = pd.DataFrame(smoothed_curves, columns=['X', 'Y', 'Z', 'PARTICLE'])
# CURVE = CURVE[CURVE.PARTICLE != -1]
# CURVE = LINKED[LINKED.PARTICLE == 28]
# CURVE = smoothed_curves_df[smoothed_curves_df.PARTICLE != -1]
# CURVE = smoothed_curves_df[smoothed_curves_df.PARTICLE == 28]

# LINKED2 = LINKED
# LINKED2['ZZ'] = LINKED['Z']*5

# fig = px.scatter_3d(LINKED, x='X', y='Y', z='Z', color='PARTICLE', size='I_GS')
# fig = px.line_3d(LINKED2, x='X', y='Y', z='ZZ', color='PARTICLE')
fig = px.line_3d(smoothed_curves_df, x='X', y='Y', z='Z', color='PARTICLE')
fig.update_traces(marker=dict(size=1))

#fig.add_trace(fig2)
plot(fig)

# fig.write_html(PATH[:-3]+'html')
# fig.write_html(PATH[:-4]+'_DBSCAN_eps5_minsamp30_smooth0999999.html')


# %%
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# fig = make_subplots(rows=1, cols=2,
#                     specs=[[{'is_3d': True}, {'is_3d': True}],
#                            ])

# fig.add_trace(go.Scatter3d(
#     x=CURVE.X, 
#     y=CURVE.Y, 
#     z=CURVE.Z,
#     mode='markers', 
#     marker=dict(
#         size=1,
#         color=CURVE['FRAME'].values,
#         colorscale='Viridis'
#         ),
#     hovertext=['X+Y+Z+FRAME'],
#     hoverinfo='all'
    
# ),row=1, col=1)

# fig.add_trace(go.Scatter3d(x=LINKED.X, y=LINKED.Y, z=LINKED.Z,
#                     mode='markers',
#                     marker=dict(
#                         size=1,
#                         color=CURVE.PARTICLE.values,
#                         colorscale='Viridis',
#                         )
# ), row=1, col=2)

# fig.show()
# plot(fig)
# %%
