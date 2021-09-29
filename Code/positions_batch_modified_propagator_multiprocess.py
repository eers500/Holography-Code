#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:21:30 2021

@author: erick
"""

#%% Import vido and set paramaters
import time
import numpy as np
import easygui
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
import easygui as gui
from multiprocessing import Pool, Process, freeze_support, set_start_method
from multiprocessing import cpu_count
from tqdm import tqdm

PATH = easygui.fileopenbox(default='F:\\PhD')
# PATH = 'MF1_30Hz_200us_awaysection.avi'
#PATH = '10x_laser_50Hz_10us_g1036_bl1602_500frames.avi'
# PATH = 'F:\\PhD\\Colloids\\20x_50Hz_100us_642nm_colloids_2000frames.avi'
T0 = time.time()
VID = f.videoImport(PATH, 0)
FRAMES_MEDIAN = 20
I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)

export_csv = True
# N = 1.3226
# LAMBDA = 0.642              # HeNe
# # MPP = 10
# FS = 0.711                     # Sampling Frequency px/um
# SZ = 5                       # # Step size um
# NUMSTEPS = 150
# THRESHOLD = 0.1

#%%  Calculate propagators, gradient stack and compute particle position ins 3D
# NUM_FRAMES = np.shape(VID)[-1]
# NUM_FRAMES = int(np.floor(np.shape(VID)[-1]/2))
NUM_FRAMES = 2000
# LOCS = np.empty((NUM_FRAMES, 3), dtype=object)
# INTENSITY = np.empty(NUM_FRAMES, dtype=object)

pbar = tqdm(total=NUM_FRAMES/cpu_count())

def my_function(TUPLE):
    I = TUPLE[0]
    I_MEDIAN = TUPLE[1]
    # I_MEDIAN = np.load('/media/erick/NuevoVol/LINUX_LAP/PhD/Colloids/MED_20x_50Hz_100us_642nm_colloids-1.npy')
    N = 1.3226
    LAMBDA = 0.642              # HeNe
    # MPP = 10
    FS = 0.711                     # Sampling Frequency px/um
    SZ = 10                       # # Step size um
    NUMSTEPS = 150
    LOCS = np.empty((1, 3), dtype=object)
    X, Y, Z, I_FS, I_GS = [], [] ,[], [], []
    # X, Y, Z, I_FS, I_GS = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
#
    # for i in range(NUM_FRAMES):
    # I = IT[i]
    # IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS).astype('float32')
    # GS = f.zGradientStack(IM).astype('float32')  
    GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
    #GS[GS < THRESHOLD] = 0
    LOCS[0, 0] = f.positions3D(GS, peak_min_distance=35)
    A = LOCS[0, 0].astype('int')
    # LOCS[0, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 1] = GS[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
        
    Y.append(LOCS[0, 0][:, 0]) 
    X.append(LOCS[0, 0][:, 1])
    Z.append(LOCS[0, 0][:, 2])
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])
    
    #print(time.time())
        
    pbar.update(1)
        
    return X, Y, Z, I_FS, I_GS

    
IT = np.empty((NUM_FRAMES), dtype=object)
MED = np.empty((NUM_FRAMES), dtype=object)
for i in range(NUM_FRAMES):
    IT[i] = VID[:, :, i]
    MED[i] = I_MEDIAN

if __name__ == "__main__":    
    # freeze_support()
    # set_start_method('spawn')
    T0 = time.time()
    pool = Pool(cpu_count())
    #pool = Pool(1)
    results = pool.map(my_function, zip(IT, MED))
    # results = pool.map_async(my_function, zip(IT, MED))
    # results = pool.imap_unordered(my_function, zip(IT, MED))
    # results = pool.apply_async(my_function, zip(IT, MED), callback=update)
    pool.close()  # 'TERM'
    # pool.join()   # 'KILL'
    T = time.time() - T0
    print('\n Elapsed time: ', T)

    POSITIONS = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'])
    for i in range(NUM_FRAMES):
      # XYZ, I_FS, I_GS, FRAME = LOCS[i, 0], LOCS[i, 1], LOCS[i, 2], i*np.ones_like(LOCS[i, 2])
            
        X, Y, Z, I_FS, I_GS, FRAME = results[i][0][0], results[i][1][0], results[i][2][0], results[i][3][0], results[i][4][0], i*np.ones_like(results[i][0][0])
        DATA = np.concatenate((np.expand_dims(X, axis=1), np.expand_dims(Y, axis=1), np.expand_dims(Z, axis=1), np.expand_dims(I_FS, axis=1), np.expand_dims(I_GS, axis=1), np.expand_dims(FRAME, axis=1)), axis=1)
        POSITIONS = POSITIONS.append(pd.DataFrame(DATA, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME']))

    if export_csv:        
        # EXPORT_PATH = PATH[:-4]+'_TH01_MPD35_multiprocess.csv'
        EXPORT_PATH = gui.filesavebox(filetypes='csv', default='.csv', msg='Save file')
        POSITIONS.to_csv(EXPORT_PATH)  # For leptospira data
	
#   3D Scatter Plot
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot
    #%matplotlib qt 

    fig = pyplot.figure()
    ax = Axes3D(fig)
    title = '3d plot'
    LOCS = POSITIONS.values
    ax.scatter(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2], s=25, marker='o')
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title(title, fontsize='20')
    ax.set_xlabel('x (pixels)', fontsize='18')
    ax.set_ylabel('y (pixels)', fontsize='18')
    ax.set_zlabel('z (slices)', fontsize='18')
    pyplot.show()
   