# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.image as mpimg
import time
import functions as f
import tkinter as tk
import tkinter.filedialog
from skimage.feature import peak_local_max

T0 = time.time()
tk.Tk().withdraw()
PATH = tk.filedialog.askopenfilename(title = "Select file",filetypes = (("avi files","*.avi"),("all files","*.*")))

VID = f.videoImport(PATH)
I_MEDIAN = f.medianImage(VID)
#I = mpimg.imread('131118-1.png')
Z =0.2*np.arange(1, 151)
THRESHOLD = 0.3
#%%
I = VID[:, :, 0]
#IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
GS, IM = f.zGradientStack(I, I_MEDIAN, Z)    # GradientStack and RS propagator
GS[GS<THRESHOLD] = 0
#%% From Labview code
#from functions import videoImport
#GS = videoImport('131118-1_(frame0)gradient.avi')

#%% Find (x,y,z) of cells
LOCS = np.zeros((1,3))
for k in range(GS.shape[2]):
   PEAKS = peak_local_max(GS[:, :, k], indices=True)   #Check for peak radius
   ZZ = np.ones((PEAKS.shape[0],1))*k
   PEAKS = np.append(PEAKS, ZZ, axis=1)
   LOCS = np.append(LOCS, PEAKS, axis=0)
LOCS = np.delete(LOCS, 0, 0)   

np.savetxt('locs.txt', LOCS)    
print(time.time()-T0)
#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(LOCS[:, 0], LOCS[:, 1], LOCS[:, 2])
pyplot.show()
    