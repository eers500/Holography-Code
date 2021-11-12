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
I = VID[:, :, 0]
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
FS = 0.711                     # Sampling Frequency px/um
SZ = 10   

rs = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, 150)
# _, BINS = np.histogram(rs.flatten())
# rs[rs < BINS[8]] = 0


# gs = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, 100)

#%% Export LUT
# path = '/media/erick/NuevoVol/LINUX_LAP/PhD/Archea_LW/LUT/'
# path = '/media/erick/NuevoVol/LINUX_LAP/PhD/Archea_LW/LUT_CES_70/'
# path = '/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/may2021/5/20x_100Hz_05us_EcoliHCB1-07/LUT/'
# path = '/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/June2021/14/sample_1/40x_HCB1_60Hz_1.259us_03/LUT/'
# path = '/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/June2021/14/sample_1/40x_HCB1_60Hz_09us_06/LUT/'
# path = '/media/erick/NuevoVol/LINUX_LAP/PhD/E_coli/June2021/14/sample_2/LUT/'
# path = '/media/erick/NuevoVol/LINUX_LAP/PhD/Archea_LW/Octypus_batch/'
path = 'C:\\Users\\eers500\\Documents\\PhD\\Colloids\\LUT\\'

ci, cj = 369, 443
D = 22

# for i in range(rs.shape[-1]):
for i in range(0, 30, 1):
    # print(i)
    plt.imsave(path+np.str(i)+'.png', rs[ci-D:ci+D, cj-D:cj+D , i], cmap='gray')   
    
    
    