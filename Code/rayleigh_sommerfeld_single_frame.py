#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:27:07 2021

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import easygui as gui

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/may2021/5/')
VID = f.videoImport(path, 0)
# VID, cdf = f.histeq(VID)
I = VID[:, :, 0] #133
# I, cdf = f.histeq(IN)

# I_MEDIAN = f.medianImage(VID, 1000)
# I_MEDIAN[I_MEDIAN == 0] = np.mean(I_MEDIAN)

# IN = I / I_MEDIAN
# IN, cdf = f.histeq(IN)

path_med = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/may2021/5/')
# I_MEDIAN = np.ones_like(I)
I_MEDIAN = plt.imread(path_med)
N = 1.3226
LAMBDA = 0.642              # HeNe
MPP = 20
FS = 0.711 * (MPP/10)                     # Sampling Frequency px/um
SZ = 2.5   

rs = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, 150, True)
# _, BINS = np.histogram(rs.flatten())
# rs[rs < BINS[8]] = 0

#%%
# path_write = gui.diropenbox()

rs_binary = np.empty_like(rs)
for i in range(rs_binary.shape[-1]):
    temp = rs[:, :, i]
    t = np.zeros_like(temp)
    t[temp >= temp.mean()] = 255
    rs_binary[:, :, i] = t
    # plt.imsave(path_write+'\\'+str(i)+'.png', rs[ci-D:ci+D, cj-D:cj+D , i], cmap='gray')
    
    
#%% Export LUT

path_write = gui.diropenbox()

ci, cj = 227, 258
D = 57
binary = True
# for i in range(rs.shape[-1]):
for i in range(0, 40, 1):
    # print(i)
    if binary:
        plt.imsave(path_write+'\\'+str(i)+'.png', rs_binary[ci-D:ci+D, cj-D:cj+D , i], cmap='gray') 
    else:
        plt.imsave(path_write+'\\'+str(i)+'.png', rs[ci-D:ci+D, cj-D:cj+D , i], cmap='gray') 
        
    
    
    