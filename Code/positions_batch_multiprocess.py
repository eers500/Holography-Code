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
import easygui as gui
from multiprocessing import Pool, Process, freeze_support, set_start_method
from multiprocessing import cpu_count
from tqdm import tqdm

#%%
# PATH = easygui.fileopenbox(default='F:\\PhD')
# PATH = 'C:\\Users\\eers500\\Documents\\PhD\\Archea_LW\\NEW ANALYSIS\\20x_ArchFM1_30Hz_150us_1_2000frames_every2_300.avi'
# PATH = 'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\NEW ANALYSIS\\40x_HCB1_60Hz_1.259us_03_430frames.avi'
# PATH = 'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1_07\\NEW ANALYSIS\\20x_100Hz_05us_EcoliHCB1-07-500.avi'
# PATH = 'C:\\Users\\eers500\\Documents\\PhD\\E_coli\\June2021\\14\\sample_1\\40x_HCB1_60Hz_1.259us_03\\NEW ANALYSIS\\40x_HCB1_60Hz_1.259us_03_BSMM_430frames.avi'

# T0 = time.time()                                       
# VID = f.videoImport(PATH, 0)
# FRAMES_MEDIAN = 20
# I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)
# # I_MEDIAN = np.ones((VID.shape[0], VID.shape[1]))

# FRAME_RATE = 60
# THRESHOLD = 0.5
# NUM_FRAMES = np.shape(VID)[-1]

#%%  Calculate propagators, gradient stack and compute particle position ins 3D

def positions_batch(TUPLE):
    I = TUPLE[0]
    I_MEDIAN = TUPLE[1]
    # I_MEDIAN = np.load('/media/erick/NuevoVol/LINUX_LAP/PhD/Colloids/MED_20x_50Hz_100us_642nm_colloids-1.npy')
    N = 1.3226
    LAMBDA = 0.642              
    MPP = 40                     # Magnification
    FS = (MPP/10)*0.711          # Sampling Frequency px/um
    SZ = 5                       # Step size um
    NUMSTEPS = 35
    THRESHOLD = 0.5
    LOCS = np.empty((1, 3), dtype=object)
    X, Y, Z, I_FS, I_GS = [], [] ,[], [], []
    IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS, True).astype('float32')
    GS = f.zGradientStack(IM).astype('float32')  
    # GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
    GS[GS < THRESHOLD] = 0
    LOCS[0, 0] = f.positions3D(GS, peak_min_distance=60, num_particles='None', MPP=MPP)
    A = LOCS[0, 0].astype('int')
    LOCS[0, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[0, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
        
    X.append(LOCS[0, 0][:, 0]*(1/FS))
    Y.append(LOCS[0, 0][:, 1]*(1/FS))
    Z.append(LOCS[0, 0][:, 2]*SZ)
    I_FS.append(LOCS[0, 1])
    I_GS.append(LOCS[0, 2])
        
    return X, Y, Z, I_FS, I_GS

    
# IT = np.empty((NUM_FRAMES), dtype=object)
# MED = np.empty((NUM_FRAMES), dtype=object)
# for i in range(NUM_FRAMES):
#     IT[i] = VID[:, :, i]
#     MED[i] = I_MEDIAN

if __name__ == "__main__":    
    # freeze_support()
    # set_start_method('spawn')
    
    PATH = easygui.fileopenbox(default='F:\\PhD')
    
    VID = f.videoImport(PATH, 0)
    FRAMES_MEDIAN = 20
    I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)
    # I_MEDIAN = np.ones((VID.shape[0], VID.shape[1]))

    FRAME_RATE = 60
    # THRESHOLD = 0.5
    NUM_FRAMES = np.shape(VID)[-1]
    
    IT = np.empty((NUM_FRAMES), dtype=object)
    MED = np.empty((NUM_FRAMES), dtype=object)
    for i in range(NUM_FRAMES):
        IT[i] = VID[:, :, i]
        MED[i] = I_MEDIAN
    
    ##
    T0 = time.time()
    pool = Pool(cpu_count())
    results = []
    for _ in tqdm(pool.imap_unordered(positions_batch, zip(IT, MED)), total=NUM_FRAMES):
        results.append(_)


    pool.close()  # 'TERM'
    #pool.join()   # 'KILL'
    T = time.time() - T0
    print('\n Elapsed time: ', T)

    export_bool = True
    # if export_bool:
    POSITIONS = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME'])
    for i in range(NUM_FRAMES):

        X = results[i][0][0]
        Y = results[i][1][0]
        Z = results[i][2][0]
        I_FS = results[i][3][0]
        I_GS = results[i][4][0]
        FRAME = i*np.ones_like(results[i][0][0])
        TIME = i*np.ones_like(results[i][0][0])
        
        DATA = np.concatenate((np.expand_dims(X, axis=1), 
                               np.expand_dims(Y, axis=1), 
                               np.expand_dims(Z, axis=1), 
                               np.expand_dims(I_FS, axis=1), 
                               np.expand_dims(I_GS, axis=1), 
                               np.expand_dims(FRAME, axis=1), 
                               np.expand_dims(FRAME*(1/FRAME_RATE), axis=1)), 
                              axis=1)
        POSITIONS = POSITIONS.append(pd.DataFrame(DATA, 
                                                  columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME', 'TIME']))
        
    if export_bool:
        # EXPORT_PATH = PATH[:-4]+'_TH01_MPD35_multiprocess.csv'
        EXPORT_PATH = gui.filesavebox(filetypes='csv', default='.csv', msg='Save file')
        POSITIONS.to_csv(EXPORT_PATH)  # For leptospira data
	

	
    

#%% Matplotlib scatter plot
# 3D Scatter Plot
# import easygui as gui
# import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
# # %matplotlib qt

# # path = gui.fileopenbox()
# # POSITIONS = pd.read_csv(path)

# X = POSITIONS.X
# Y = POSITIONS.Y
# Z = POSITIONS.Z
# T = POSITIONS.TIME

# # X = LINKED.X
# # Y = LINKED.Y
# # Z = LINKED.Z
# # T = LINKED.PARTICLE

# fig = pyplot.figure()
# # ax = Axes3D(fig)
# ax = pyplot.axes(projection='3d')

# ax.scatter(X, Y, Z, s=5, marker='o', c=T)
# ax.tick_params(axis='both', labelsize=10)
# ax.set_title('Cells Positions in 3D', fontsize='20')
# ax.set_xlabel('x (um)', fontsize='18')
# ax.set_ylabel('y (um)', fontsize='18')
# ax.set_zlabel('z (um)', fontsize='18')

# pyplot.show()

    

# %%
