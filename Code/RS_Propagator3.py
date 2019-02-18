#%% -*- coding: utf-8 -*-
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import matplotlib.image as mpimg
from scipy import signal
import numpy as np
import scipy
from scipy import ndimage


I = mpimg.imread('131118-1.png')
#img = mpimg.imread('MF1_30Hz_200us_away_median.png')
#%% Median image
IB = signal.medfilt2d(I, kernel_size = 3)
iz = np.where(IB == 0)
IB[IB == 0] = np.median(IB)

IN = I/IB

from functions import Bandpass_Filter
_,BP = Bandpass_Filter(IN,30,120)

FT = np.fft.fft2(IN-1)

n = 1.3326
lambdaa = 0.532
N = np.shape(IN)[0]
fs = np.arange(N)/N - 0.5
z = np.arange(1,21)

#%%
for ii in range(N):
    for jj in range(N):
        P[i,j] = (lambdaa*)