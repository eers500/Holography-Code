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
from tqdm import tqdm

PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
# DF = pd.read_csv(PATH, index_col=1)
DF = pd.read_csv(PATH)
DF = DF[['X','Y','Z','I_FS','I_GS','FRAME', 'TIME']]
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
    DBSCAN = cl.DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=cores).fit(DF[['X', 'Y', 'Z']])
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
    TR = pd.DataFrame(TRACK, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME'])
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


#%% MSD
particle_num = np.unique(LINKED['PARTICLE'])

pp = []
for p in particle_num:
    
    L = LINKED[LINKED['PARTICLE'] ==  p]
    if len(L) > 50:
        temp = f.clean_tracks(L)
        L = pd.DataFrame.transpose(pd.DataFrame(temp, ['X', 'Y', 'Z', 'TIME', 'FRAME','PARTICLE']))
       
        _, swim = f.MSD(L.X.values, L.Y.values, L.Z.values)
        print(swim)
         
        if swim == False:
            LINKED = LINKED[LINKED['PARTICLE'] != p]
      
    else:
        LINKED = LINKED[LINKED['PARTICLE'] != p]

#%% Smooth trajectories
spline_degree = 3  # 3 for cubic spline
particle_num = np.sort(LINKED.PARTICLE.unique())
T0_smooth = time.time()
smoothed_curves = -np.ones((1, 5))
for pn in tqdm(particle_num):
    # Do not use this
    # L = LINKED[LINKED.PARTICLE == pn].values
    # X = f.smooth_curve(L, spline_degree=spline_degree, lim=20, sc=3000)
    
    L = LINKED[LINKED.PARTICLE == pn]
    temp = f.clean_tracks(L)
    L = pd.DataFrame.transpose(pd.DataFrame(temp, ['X', 'Y', 'Z', 'TIME', 'FRAME','PARTICLE']))

    if len(L) < 100:
        continue
    X = f.csaps_smoothing(L, smoothing_condition=0.999, filter_data=False, limit=5)
    
    if X != -1:
        smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], X[3], pn*np.ones_like(X[1])), axis=1))) 

smoothed_curves = smoothed_curves[1:, :]
smoothed_curves_df = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'TIME','PARTICLE'])
T_smooth = time.time() - T0_smooth

# smoothed_curves_df.to_csv(PATH[:-4]+'_DBSCAN_smooth_200.csv', index=False)


#%% Matplotlib scatter plot to compare detected points with smoothed curve
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
%matplotlib qt

p_number = 0
# CURVE_1 = LINKED[LINKED.PARTICLE == p_number]
CURVE_1 = LINKED
# CURVE_2 = smoothed_curves_df[smoothed_curves_df.PARTICLE == p_number]
CURVE_2 = smoothed_curves_df



fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(CURVE_1.X, CURVE_1.Y, CURVE_1.Z, 'r.', label='Detected Positions', c=np.arange(len(CURVE_1.X)), alpha=0.3)
# p = ax.scatter(CURVE_1.X, CURVE_1.Y, CURVE_1.Z, 'r.', label='Detected Positions', c=CURVE_1.TIME, alpha=0.3)

# ax.scatter(CURVE_2.X, CURVE_2.Y, CURVE_2.Z, label='Detected Positions', c=np.arange(len(CURVE_2.X)), s=30, marker='.', alpha=0.3)
# # ax.plot(CURVE_1.X, CURVE_2.Y, CURVE_2.Z, 'r-', label='Smoothed Curve')
# ax.plot(CURVE_2.X, CURVE_2.Y, CURVE_2.Z, 'r-', label='Smoothed Curve')
# # # ax.plot(CURVE_2.X[CURVE_2.X>350], CURVE_2.Y[CURVE_2.X>350], CURVE_2.Z[CURVE_2.X>350], 'r-', label='Smoothed Curve')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_zlim(bottom=0, top=40)
# fig.colorbar(p)
ax.set_title('Smoothed Track')
pyplot.show()

#%% Crete Data Frame with Speed
particle_num = np.unique(smoothed_curves_df['PARTICLE'])

xx, yy, zz, tt, pp, sp = -1, -1, -1, -1, -1, -1

for pn in particle_num:
    s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    # print(pn, len(s))

    if len(s) > 100:
        speed, x, y, z, t = f.get_speed(s)
        xx = np.hstack((xx, x))
        yy = np.hstack((yy, y))
        zz = np.hstack((zz, z))
        tt = np.hstack((tt, t))
        pp = np.hstack((pp, pn*np.ones(len(t))))
        sp = np.hstack((sp, speed))
    

tracks_w_speed = pd.DataFrame(np.transpose([xx[1:], yy[1:], zz[1:], tt[1:], pp[1:], sp[1:]]), columns=['X', 'Y', 'Z', 'TIME', 'PARTICLE', 'SPEED'])

# PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', filetypes='.csv')
# tracks_w_speed = pd.read_csv(PATH, index_col=False)

fig = plt.figure(1, dpi=150)
ax = fig.add_subplot(111, projection='3d')


# p = ax.scatter(tracks_w_speed['Y'], tracks_w_speed['X'], tracks_w_speed['Z'], c=tracks_w_speed['SPEED'], marker='.', s=20)
# cbar = plt.colorbar(p)
# cbar.set_label('Speed ($\mu ms^{-1}$)')

for pn in particle_num:
    # s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    s = tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
    ax.plot(s['X'], s['Y'], s['Z'], linewidth=2)
    # ax.scatter(s['Y'], s['X'], s['Z'])

ax.axis('tight')
ax.set_title('$\it{Escherichia \ Coli}$', fontsize=40)  # $\it{Escherichia \ Coli}$
ax.set_xlabel('y ($\mu$m)', fontsize=20)
ax.set_ylabel('x ($\mu$m)', fontsize=20)
ax.set_zlabel('-z ($\mu$m)', fontsize=20)
# ax.set_zlim(bottom=0, top=40)

plt.figure(2)
plt.hist(tracks_w_speed['SPEED'], 13)
mean_speed = tracks_w_speed['SPEED'].mean()
print(mean_speed)
plt.title('Speed: $\mu$ = ' + str(np.float16(mean_speed)) + ' $\mu m s^{-1}$', fontsize=40)
plt.xlabel('Speed ($\mu m s^{-1}$)', fontsize=20)
plt.ylabel('Frequency', fontsize=20)

pyplot.show()

#%% Speed
# %matplotlib qt
# plt.rcParams['figure.dpi'] = 150 

# PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', filetypes='.csv')
# smoothed_curves_df = pd.read_csv(PATH,index_col=False)
particle_num = np.unique(smoothed_curves_df['PARTICLE'])

fig = plt.figure(2, dpi=150)
ax = fig.add_subplot(111, projection='3d')

for pn in particle_num:
    s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    # speed, x, y, z = f.get_speed(s)
    # ax.scatter(x, y, z, c=speed, marker='.', s=5)
    # ax.scatter(LINKED.Y, LINKED.X, LINKED.Z, '.', s=1)
    b = s[s['Z'] < 40]
    ax.plot(0.7*b.X, 0.7*b.Y, 1.2*b.Z, linewidth=2)
    # ax.plot(0.7*s.X, 0.7*s.Y, s.Z, linewidth=2)
    
# p = ax.scatter(s['X'][:-1], s['Y'][:-1], s['Z'][:-1], c=vv, s=2)
# fig.colorbar(p)
# cbar = plt.colorbar(p)
# cbar.set_label('Speed ($\mu ms^{-1}$)')

ax.axis('tight')
# ax.set_title('Archea', fontsize=40)  # $\it{Escherichia \ Coli}$
ax.set_xlabel('y ($\mu$m)', fontsize=20)
ax.set_ylabel('x ($\mu$m)', fontsize=20)
ax.set_zlabel('-z ($\mu$m)', fontsize=20)
ax.set_zlim(bottom=0, top=40)

pyplot.show()

#%% Matplotlib scatter plot
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

# PKS = A.__array__()
# np.savetxt('locs.txt', PKS)
fig = pyplot.figure()
ax = Axes3D(fig)

X = LINKED.X
Y = LINKED.Y
Z = LINKED.Z
T = LINKED.TIME
P = LINKED.PARTICLE


# ax.scatter(X, Y, Z, s=2, marker='o', c=P)
ax.plot(X, Y, Z)
# ax.plot(smoothed_curves_df.Y, smoothed_curves_df.X, smoothed_curves_df.Z, 'r-')
ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (um)', fontsize='18')
ax.set_ylabel('y (um)', fontsize='18')
ax.set_zlabel('z (um)', fontsize='18')
# fig.colorbar(p, ax=ax)
pyplot.show()
    
    


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
fig = px.line_3d(smoothed_curves_df, x='X', y='Y', z='Z', color='PARTICLE', hover_data=['TIME'])
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
