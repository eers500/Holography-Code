#%% -*- coding: utf-8 -*-
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import matplotlib.image as mpimg
from scipy import signal
import numpy as np
import math as m

img = mpimg.imread('MF1_30Hz_200us_away_median.png')

#%% Normalize with median image ND FFT
imgFILT = signal.medfilt(img, kernel_size = 3)
hologram = imgFILT-1   

Efft = np.fft.fft2(hologram)
                      # b(r)-1
size = np.shape(hologram)
nx = size[0]
ny = size[1]

#%%
lambdaa = 0.525                                  # Wavelength of field
nm = 1                                           # Refraction index of medium
k = 2*m.pi/lambdaa                               # wavenumber 

x = np.arange(nx, dtype = np.float)
y = np.arange(ny, dtype = np.float)
z = np.arange(1,10, dtype = np.float)

xx,yy,zz = np.meshgrid(x,y,z)

#%% Rayleigh-Sommerfeld Back-Propagator
Hz = lambda x,y,z: (2*m.pi)**(-1)*z*(np.exp(1j*k*(x**2+y**2+z**2)**(1/2)))*(1j*k*(x**2+y**2+z**2)**(-1)-(x**2+y**2+z**2)**(-3/2))
HZ = Hz(xx,yy,zz)

EE = np.empty([nx,ny,z.shape[0]], dtype=np.complex)
for ii in range(z.shape[0]):
    EE[:,:,ii] = Efft[:,:]*HZ[:,:,ii]
    E = np.abs(np.fft.fft2(EE))

#%% 
import matplotlib.pyplot as plt
#
plt.figure(1)
plt.imshow(np.log10(E[:,:,8]), cmap = 'gray')
