# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.image as mpimg
#import time
import functions as f
import tkinter as tk
import tkinter.filedialog

tk.Tk().withdraw()
PATH = tk.filedialog.askopenfilename()

VID = f.videoImport(PATH)
I_MEDIAN = f.medianImage(VID)
#I = mpimg.imread('131118-1.png')
Z = 0.02*np.arange(1, 151)

#%% Batch calculation
FILENAME = "frameStack_frame"
for i in range(3):
#for i in range(VID.shape[2]):
    FILE = FILENAME + str(i)
    IM = f.rayleighSommerfeldPropagator(VID[:, :, i], I_MEDIAN, Z)
    f.exportAVI(FILE+".avi", IM, IM.shape[0], IM.shape[1], 24)
#%%
I = VID[:, :, 0]
IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
GS = f.zGradientStack(I, I_MEDIAN, Z)


f.exportAVI('frameStack.avi',IM, IM.shape[0], IM.shape[1], 24)


VI = f.videoImport('frameStack.avi')

