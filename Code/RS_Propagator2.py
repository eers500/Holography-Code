#%% -*- coding: utf-8 -*-
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import matplotlib.image as mpimg
from scipy import signal
import numpy as np

img = mpimg.imread('MF1_30Hz_200us_away_median.png')
#%%
I0 = signal.medfilt2d(img, kernel_size = 3)
I0[I0 == 0] = np.median(I0) 

b = np.complex128(img/I0-1)

size = np.shape(b)
nx = size[0]
ny = size[1]

lambdaa = 0.525                                  # Wavelength of field
nm = 1                                           # Refraction index of medium
k = 2*m.pi/lambdaa                               # wavenumber 

x = np.arange(nx, dtype = np.float)
y = np.arange(ny, dtype = np.float)
z = np.arange(1,10, dtype = np.float)

xx,yy,zz = np.meshgrid(x,y,z)

Hz = lambda x,y,z: (2*m.pi)**(-1)*z*(np.exp(1j*k*(x**2+y**2+z**2)**(1/2)))*\
                    (1j*k*(x**2+y**2+z**2)**(-1)-(x**2+y**2+z**2)**(-3/2))
HZ = Hz(xx,yy,zz)


    
    
    
    
    
    
    
    
    
    
    