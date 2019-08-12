# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.image as mpimg
import time
import functions as f
# import tkinter as tk
# from  tkinter.filedialog import askopenfilename
import easygui
from skimage.feature import peak_local_max

# tk.Tk().withdraw()
# PATH = tk.filedialog.askopenfilename(title="Select file", filetypes=(("avi files", "*.avi"), ("all files", "*.*")))
PATH = easygui.fileopenbox('Select a recording file', 'Recording', filetypes=['*.avi'])

# T0 = time.time()
VID = f.videoImport(PATH, 0)
T0 = time.time()
I_MEDIAN = f.medianImage(VID, 20)
# I_MEDIAN = mpimg.imread('MED_DMEM_s1_10x_50Hz_50us_away5-004-1.png')
# I = mpimg.imread('131118-1.png')
Z = np.arange(1, 151)
# THRESHOLD = 0.3
T = time.time()-T0
print(T)
#%%
I = VID[:, :, 0]
# IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
GS, IM = f.zGradientStack(I, I_MEDIAN, Z)  # GradientStack and RS propagator
THRESHOLD = 0.3
GS[GS < THRESHOLD] = 0

#%%
# From Labview code
# from functions import videoImport
# GS = videoImport('131118-1_(frame0)gradient.avi')

#%%
# Find (x,y,z) of cells
LOCS = np.zeros((1, 3))
for k in range(GS.shape[2]):
   PEAKS = peak_local_max(GS[:, :, k], indices=True)  # Check for peak radius
   ZZ = np.ones((PEAKS.shape[0], 1)) * k
   PEAKS = np.append(PEAKS, ZZ, axis=1)
   LOCS = np.append(LOCS, PEAKS, axis=0)
LOCS = np.delete(LOCS, 0, 0)

np.savetxt('locs.txt', LOCS)
print(time.time() - T0)

#%%
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2], s=25, marker='o')
ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')
pyplot.show()
