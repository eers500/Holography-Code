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
k = 2*m.pi*n/lambdaa
N = np.shape(IN)[0]
nx = np.matrix(np.arange(N)/N - 0.5)
nx = np.tile(nx,(N,1))
ny = np.transpose(nx)
#ny = np.tile(ny,(1,N))
fs = 1
z = np.arange(1,21)

#%%
#qsq = ((lambdaa/(N*n))*nx)**2 + ((lambdaa/(N*n))*ny)**2
P = np.empty_like(IB)
for i in range(N):
    for j in range(N):
#        print(i,j)
        P[i,j] = ((lambdaa*fs)/(N*n))**2*((i-N/2)**2+(j-N/2)**2)