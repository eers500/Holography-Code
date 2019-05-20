# -*- coding: utf-8 -*-
#%% 
import time
import tkinter as tk
import tkinter.filedialog
import numpy as np
#import matplotlib.image as mpimg
import functions as f


tk.Tk().withdraw()
PATH = tk.filedialog.askopenfilename()

VID = f.videoImport(PATH)
I_MEDIAN = f.medianImage(VID)
import scipy.misc
scipy.misc.imsave('AVG_20-02-19_ECOLI_HCB1_100Hz_290us_10x_1.png', I_MEDIAN)
#I = mpimg.imread('131118-1.png')
Z = 0.02*np.arange(1, 151)

#%% Batch calculation
FILENAME = "frameStack_frame"
T0 = time.time()
T = np.empty(3)
for i in range(3):
#for i in range(VID.shape[2]):
    FILE = FILENAME + str(i)
    IM = f.rayleighSommerfeldPropagator(VID[:, :, i], I_MEDIAN, Z)
    f.exportAVI(FILE+".avi", IM, IM.shape[0], IM.shape[1], 24)
    T[i] = time.time() - T0
    print(T[i])
#%%
#I = VID[:, :, 0]
#IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
#GS = f.zGradientStack(I, I_MEDIAN, Z)
#f.exportAVI('frameStack.avi', IM, IM.shape[0], IM.shape[1], 24)
#
#VI = f.videoImport('frameStack.avi')
