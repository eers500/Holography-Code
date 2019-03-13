# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.image as mpimg
#import time
import functions as fc

VID = fc.videoImport('131118-1.avi')
I_MEDIAN = fc.medianImage(VID)


#I = mpimg.imread('131118-1.png')
Z = 0.02*np.arange(1, 151)
I = VID[:, :, 0]
IM = fc.rayleighSommerfeldPropagator(I, I_MEDIAN, Z)


fc.exportAVI('frameStack.avi',IM, IM.shape[0], IM.shape[1], 24)


VI = fc.videoImport('frameStack.avi')

