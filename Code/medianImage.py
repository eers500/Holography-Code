# -*- coding: utf-8 -*-
import numpy as np
import time
from functions import videoImport

#T0 = time.time()
VID = videoImport('131118-1.avi')

#%%
T0 = time.time()

MEAN = np.median(VID, axis=2)

T = time.time()-T0
