#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:44:28 2020

@author: erick
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.image as mpimg
import functions as f


# NAMES = glob.glob('/media/erick/NuevoVol/LINUX_LAP/PhD/200812_GT/*.png')
NAMES = glob.glob('/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/PNG/*.png')

PATH = os.path.split(NAMES[1])[0]+'/'

NAME = []
for i in range(len(NAMES)):
    # NAME.append(NAMES[i][45:-4])
    # NAME[i] = NAMES[0][42:-4]
    NAME.append(os.path.split(NAMES[i])[1][:-4])
    
    
N = 1.3226
LAMBDA = 0.642              # red 642 green 520
MPP = 10
FS = 0.711                     # Sampling Frequency px/um
SZ = 5  
NUMSTEPS = 300
I_MEDIAN = np.ones((1024, 1024))

RS = np.empty((1024, 1024, NUMSTEPS))
T0 = time.time()
for k in range(len(NAMES)):
    I = mpimg.imread(NAMES[k])
    # f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)
    RS = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)
    
    
    IM = (RS - np.min(RS))*(255/(np.max(RS)-np.min(RS)))
    IMM = np.uint8(IM)
    # EX_PATH, NAME = os.path.split(PATH[0])
    # exportAVI(EX_PATH+NAME[0:-4]+'_frame_stack_'+str(SZ)+'um.avi', IMM, IMM.shape[0], IMM.shape[1], 30)
    
    # EX_PATH, NAME = os.path.split(PATH[0])
    f.exportAVI(NAMES[k][:-4]+'_frame_stack_'+np.str(NUMSTEPS)+'steps_'+str(SZ)+'um.avi', IMM, IMM.shape[0], IMM.shape[1], 30)
    print(time.time() - T0)
        
    