#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:27:33 2020

@author: erick
"""
#%% Import
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



#%% Run track detection
# KMeans, DBSCAN or HA
# method = 'DBSCAN'
method = gui.choicebox(msg='Choose a tracking Algorithm', title='Choose', choices=['KMeans', 'DBSCAN', 'HA'])


if method == 'KMeans':

    #% K-Means for track detection
    T0_kmeans = time.time()
    kmeans = cl.KMeans(n_clusters=D[D[:, 5] == 0].shape[0], init='k-means++').fit(DF[['X', 'Y', 'Z']])
    DF['PARTICLE'] = kmeans.labels_
    LINKED = DF
    T_kmeans = time.time() -  T0_kmeans

elif method == 'DBSCAN':

    #% Density Based Spatial Clustering (DBSC)
    T0_DBSCAN = time.time()
    DBSCAN = cl.DBSCAN(eps=5, min_samples=8).fit(DF[['X', 'Y', 'Z']])
    DF['PARTICLE'] = DBSCAN.labels_
    LINKED = DF
    L = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])
    T_DBSCAN = time.time() - T0_DBSCAN

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
        
        if k==0:    
            R1 = FRAME_DATA[k]
            R2 = FRAME_DATA[k+1]
        else:
            R1 = TRACKS[k]
            R2 = FRAME_DATA[k+1]
        
        # COST = np.empty((int(SIZE.max()), int(SIZE.max())))
        COST = np.empty((len(R1), len(R2)))
        # for i in range(int(SIZE.max())):
        #     for j in range(int(SIZE.max())):        
                
        for i in range(len(R1)):
            for j in range(len(R2)):
                
                COST[i, j] = np.sqrt(sum((R1[i, :3]- R2[j, :3])**2))      
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
# from scipy import interpolate
# from mpl_toolkits.mplot3d import Axes3D
# from scipy import ndimage

# L = LINKED[LINKED.PARTICLE == 4].values

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
    X = f.csaps_smoothing(L, smoothing_condition=0.999999, smooth_data=True)
    
    if X != -1:
        smoothed_curves = np.vstack((smoothed_curves, np.stack((X[0], X[1], X[2], pn*np.ones_like(X[1])), axis=1))) 

smoothed_curves = smoothed_curves[1:, :]
smoothed_curves_df = pd.DataFrame(smoothed_curves, columns=['X', 'Y' ,'Z', 'PARTICLE'])
T_smooth = time.time() - T0_smooth

# smoothed_curves_df.to_csv(PATH[:-4]+'_DBSCAN_smooth_200.csv', index=False)

#%% Ramer-Douglas-Peucker Algorithm to detect turns
from rdp import rdp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# L = LINKED[LINKED.PARTICLE == 6]
L = smoothed_curves_df[smoothed_curves_df.PARTICLE == 9]
xx, yy, zz = L.X.values, L.Y.values, L.Z.values
xx = xx.reshape(len(xx), 1)
yy = yy.reshape(len(yy), 1)
zz = zz.reshape(len(zz), 1)

t = np.linspace(0, 1, len(xx))
t = t.reshape(len(t), 1)

stack =  np.hstack((xx, yy, zz))

eps = 0.03
eps3d = 2
xyz = rdp(stack, epsilon=eps3d)
x = rdp(np.hstack((t, xx)), epsilon=eps)
y = rdp(np.hstack((t, yy)), epsilon=eps)
z = rdp(np.hstack((t, zz)), epsilon=eps)

fig = plt.figure(figsize=(7, 4.5))
ax0 = plt.subplot2grid((6,6), (0, 0), 2, 3)
ax0.plot(t, xx, 'b-')
ax0.plot(x[:, 0], x[:, 1], 'r.')
ax0.set_ylabel('x')
ax0.set_title('eps = '+np.str(eps))

ax1 = plt.subplot2grid((6, 6), (2, 0), 2, 3)
ax1.plot(t, yy, 'b-')
ax1.plot(y[:, 0], y[:, 1], 'r.')
ax1.set_ylabel('y')

ax2 = plt.subplot2grid((6, 6), (4, 0), 2, 3)
ax2.plot(t, zz, 'b-')
ax2.plot(z[:, 0], z[:, 1], 'r.')
ax2.set_ylabel('z')

ax3 = plt.subplot2grid((6, 6), (0, 3), 3, 4, projection='3d')
ax3.set_facecolor('none')
ax3.plot(L.X, L.Y, L.Z, 'b-', label='original data')
ax3.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'r.', label='Turns')
ax3.set_title('Smoothed track')

ax4 = plt.subplot2grid((6, 6), (3, 3), 3, 4, projection='3d')
ax4.set_facecolor('none')
ax4.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', label='RDP reconstructed eps = '+np.str(eps3d))
ax4.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'r.', label='Turns')
ax4.set_title('RDP reconstruction')
ax4.legend(loc='best')

# Use derivatives of r (distance to origin) to sense changes in direction
from scipy.signal import find_peaks
r = np.sqrt(L.X.values**2 + L.Y.values**2 + L.Z.values**2)
fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(r)

diff = np.diff(r)
diff2 = np.diff(diff)
peaks, _ = find_peaks(diff, height=0.04)
peaks2, _ = find_peaks(diff2, height=np.min(diff2)-1)   # to get all local maxima

axx = fig.add_subplot(312)
axx.plot(np.arange(len(diff)), diff)
axx.plot(peaks, diff[peaks], '.')

axx = fig.add_subplot(313)
axx.plot(np.arange(len(diff2)), diff2)
axx.plot(peaks2, diff2[peaks2], '.')

fig = plt.figure()
axxx = fig.add_subplot(121, projection='3d')
axxx.plot(L.X, L.Y, L.Z, 'b-', label='original data')
axxx.plot(L.X.values[peaks], L.Y.values[peaks], L.Z.values[peaks], 'r.')

axxx = fig.add_subplot(122, projection='3d')
axxx.plot(L.X, L.Y, L.Z, 'b-', label='original data')
axxx.plot(L.X.values[peaks2], L.Y.values[peaks2], L.Z.values[peaks2], 'r.')

plt.show()

#%% Matplotlib scatter plot to compare detected points with smoothed curve
# 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
# #%matplotlib qt

# p_number = 106
# CURVE_1 = LINKED[LINKED.PARTICLE == p_number]
# CURVE_2 = smoothed_curves_df[smoothed_curves_df.PARTICLE == p_number]

# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(CURVE_1.X, CURVE_1.Y, CURVE_1.Z, 'r.', label='Detected Positions', c=np.arange(len(CURVE_1.X)))
# ax.plot(CURVE_2.X, CURVE_2.Y, CURVE_2.Z, 'r-', label='Smoothed Curve')
# pyplot.show()


#%% Plotly scatter plot
import plotly.express as px
import pandas as pd
from plotly.offline import plot

# For Archea data that looks messy
# value_counts = LINKED.PARTICLE.value_counts()
# values = value_counts[value_counts.values < 2000]
# idx = LINKED.PARTICLE.isin(values.index)
# CURVE = LINKED.iloc[idx.values]

# CURVE = LINKED[LINKED.PARTICLE == 4]
# CURVE = LINKED[(LINKED.PARTICLE != -1)]
# CURVE = pd.DataFrame(smoothed_curves, columns=['X', 'Y', 'Z', 'PARTICLE'])
# CURVE = CURVE[CURVE.PARTICLE != -1]
# CURVE = LINKED[LINKED.PARTICLE == 28]
CURVE = smoothed_curves_df[smoothed_curves_df.PARTICLE != -1]
# CURVE = smoothed_curves_df[smoothed_curves_df.PARTICLE == 28]

# fig = px.scatter_3d(CURVE, x='X', y='Y', z='Z', color='PARTICLE')
fig = px.line_3d(CURVE, x='X', y='Y', z='Z', color='PARTICLE')
fig.update_traces(marker=dict(size=1))
plot(fig)

# fig.write_html(PATH[:-3]+'html')
# fig.write_html(PATH[:-4]+'_HA.html')


