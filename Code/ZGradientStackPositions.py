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

#%%
I = VID[:, :, 0]
IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, Z)
GS = f.zGradientStack(I, I_MEDIAN, Z)

#%%
GS50 = GS[:, :, 50]
GRAD = np.gradient(GS50)
GRADasb = np.sqrt(GRAD[0]**2+GRAD[1]**2)