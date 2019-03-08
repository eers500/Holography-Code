# -*- coding: utf-8 -*-
import numpy as np
from functions import videoImport

VID = videoImport('131118-1.avi')

#%%
MEAN_IM = np.cumsum(VID, axis=2)/VID.shape[2]

