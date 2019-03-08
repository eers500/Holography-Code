# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.image as mpimg
import time
from functions import rayleighSommerfeldPropagator, exportAVI

I = mpimg.imread('131118-1.png')
I_median = mpimg.imread('AVG_131118-2.png')
Z = 0.02*np.arange(1, 151)

T0 = time.time()
IM = rayleighSommerfeldPropagator(I, I_median, Z)


exportAVI('frameStack.avi',IM, IM.shape[0], IM.shape[1], 24)

#%%
from functions import videoImport

VI = videoImport('frameStack.avi')

