#%% -*- coding: utf-8 -*-
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import matplotlib.image as mpimg
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import time

t0 = time.time()
I = mpimg.imread('131118-1.png')
#I = mpimg.imread('img_gs.png')
#img = mpimg.imread('MF1_30Hz_200us_away_median.png')
#%% Median image
IB = mpimg.imread('AVG_131118-1.png')
#IB = mpimg.imread('img_gs.png')
#IB = signal.medfilt2d(I, kernel_size = 3)
iz = np.where(IB == 0)
IB[IB == 0] = np.average(IB)

IN = I/IB

from functions import Bandpass_Filter
_,BP = Bandpass_Filter(IN,2,30)

#FT = np.fft.fft2(IN-1)
nx = I.shape[0]
ny = I.shape[1]

n = 1.3226
lambdaa = 0.642           #HeNe
fs = 1.422                #Sampling Frequency px/um
Ni = np.shape(IN)[0]
Nj = np.shape(IN)[1]
z = 0.01*np.arange(1,1001)
K = 2*m.pi*n/lambdaa      #Wavenumber

#%%
#qsq = ((lambdaa/(N*n))*nx)**2 + ((lambdaa/(N*n))*ny)**2
P = np.empty_like(IB, dtype = complex)
for i in range(Ni):
    for j in range(Nj):
#        print(i,j)
        P[i,j] = ((lambdaa*fs)/(max([Ni,Nj])*n))**2*((i-Ni/2)**2+(j-Nj/2)**2)
        
Q = np.sqrt(1-P)-1
Q = np.conj(Q)

R = np.empty([Ni,Nj,z.shape[0]], dtype = complex)
Iz = R
for k in range(z.shape[0]):
    for i in range(Ni):
        for j in range(Nj):
            R[i,j,k] = np.exp((-1j*K*z[k])*Q[i,j])
#            R[i,j,k] = np.exp(Q[i,j])

    Iz[:,:,k] = 1+np.fft.ifft2(BP*(np.fft.fft2(IN-1))*R[:,:,k])
        
Iz = np.real(Iz)
Im = (20/np.std(Iz))*(Iz - np.mean(Iz)) + 128

t = time.time() - t0           


#plt.imshow(Im[:,:,15], cmap = 'gray')
#plt.colorbar()
#

#%%
for i in range(np.shape(z)[0]):
    plt.imshow(Im[:,:,i], cmap = 'gray')
    plt.savefig('{}.png'.format(i+1))
#    plt.clf()







