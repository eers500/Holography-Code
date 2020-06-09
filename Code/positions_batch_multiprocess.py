#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:17:55 2020

@author: erick
"""

#%% Import vido and set paramaters
import time
import numpy as np
import easygui
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
from multiprocessing import Pool
from multiprocessing import cpu_count

# PATH = easygui.fileopenbox()
# PATH = 'MF1_30Hz_200us_awaysection.avi'
#PATH = '10x_laser_50Hz_10us_g1036_bl1602_500frames.avi'
PATH = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames.avi'
T0 = time.time()
VID = f.videoImport(PATH, 0)
# FRAMES_MEDIAN = 20
# I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)

# N = 1.3226
# LAMBDA = 0.642              # HeNe
# # MPP = 10
# FS = 0.711                     # Sampling Frequency px/um
# SZ = 5                       # # Step size um
# NUMSTEPS = 150
# THRESHOLD = 1

#%%  Calculate propagators, gradient stack and compute particle position ins 3D
# NUM_FRAMES = np.shape(VID)[-1]
# NUM_FRAMES = int(np.floor(np.shape(VID)[-1]/2))
NUM_FRAMES = 2
# LOCS = np.empty((NUM_FRAMES, 3), dtype=object)
# INTENSITY = np.empty(NUM_FRAMES, dtype=object)


def my_function(IT):
    I_MEDIAN = np.load('/home/erick/Documents/PhD/Holography-Code/Code/MED_20x_50Hz_100us_642nm_colloids-1.npy')
    N = 1.3226
    LAMBDA = 0.642              # HeNe
    # MPP = 10
    FS = 0.711                     # Sampling Frequency px/um
    SZ = 5                       # # Step size um
    NUMSTEPS = 150
    LOCS = np.empty((NUM_FRAMES, 3), dtype=object)
    X, Y, Z, I_FS, I_GS = [], [] ,[], [], []
    # X, Y, Z, I_FS, I_GS = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    for i in range(NUM_FRAMES):
        I = IT[i]
        IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS).astype('float32')
        # GS = f.zGradientStack(IM).astype('float32')  
        GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
        # GS[GS < THRESHOLD] = 0.003
        LOCS[i, 0] = f.positions3D(GS, peak_min_distance=20)
        A = LOCS[i, 0].astype('int')
        LOCS[i, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
        LOCS[i, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
        
        # X = np.concatenate((X, LOCS[i, 0][:, 0]), axis=0)
        # Y = np.concatenate((Y, LOCS[i, 0][:, 1]), axis=0)
        # Z = np.concatenate((Z, LOCS[i, 0][:, 2]), axis=0)
        # I_FS = np.concatenate((I_FS, LOCS[i, 1]), axis=0)
        # I_GS = np.concatenate((I_GS, LOCS[i, 2]), axis=0)
        
        X.append(LOCS[i, 0][:, 0]) 
        Y.append(LOCS[i, 0][:, 1])
        Z.append(LOCS[i, 0][:, 2])
        I_FS.append(LOCS[i, 1])
        I_GS.append(LOCS[i, 2])
        
        # np.stack((X, Y, Z, I_FS, I_GS), axis=1)
    
        
    return X, Y, Z, I_FS, I_GS
    
IT = np.empty((NUM_FRAMES), dtype=object)
for i in range(NUM_FRAMES):
    IT[i] = VID[:, :, i]

if __name__ == "__main__":
    T0 = time.time()
    pool = Pool(cpu_count())
    results = pool.map(my_function, IT)
    pool.close()  # 'TERM'
    pool.join()   # 'KILL'
    T = time.time() - T0

#%% Matplotlib scatter plot
# 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot

# # PKS = A.__array__()
# # np.savetxt('locs.txt', PKS)
# fig = pyplot.figure()
# ax = Axes3D(fig)

# p = ax.scatter(X, Y, Z, s=25, marker='o', c=Z)
# ax.tick_params(axis='both', labelsize=10)
# ax.set_title('Cells Positions in 3D', fontsize='20')
# ax.set_xlabel('x (pixels)', fontsize='18')
# ax.set_ylabel('y (pixels)', fontsize='18')
# ax.set_zlabel('z (slices)', fontsize='18')
# fig.colorbar(p)
# pyplot.show()
    
    
