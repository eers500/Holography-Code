#%% -*- coding: utf-8 -*-
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import matplotlib.image as mpimg
from scipy import signal
import numpy as np

img = mpimg.imread('MF1_30Hz_200us_away_median.png')

#%% Normalize with median image
imgFILT = signal.medfilt(img, kernel_size = 3)
hologram = imgFILT-1   
                      # b(r)-1
size = np.shape(hologram)
nx = size[0]
ny = size[1]

#%%
lambdaa = 0.525                                  # Wavelength of field
nm = 1                                           # Refraction index of medium
k = 2*math.pi/lambdaa                                # wavenumber 

x = np.arange(nx)
y = np.arange(ny)
z = np.arange(0,10)

xx,yy = np.meshgrid(x,y)
r = (xx**2+yy**2)**(1/2)

#%% Simbolic expresion for A (differential in propagator)
from sympy import *

A = z*m.exp(1j*k*(r**2+z**2)**(1/2))*(1j*k*(r**2+z**2)**(-1)-(r**2+z**2)**(-3/2))

x1,y1,z1 = symbols('x1 y1 z1')
A = 


for kk in range(len(z)):
    
 




#%% 
#import matplotlib.pyplot as plt
#
#plt.figure(1)
#plt.imshow(imgFILT, cmap = 'gray')
#
#plt.figure(2)
#plt.imshow(imgFILTs, cmap = 'gray')
