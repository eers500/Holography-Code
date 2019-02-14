#%% -*- coding: utf-8 -*-
# Rayleigh-Sommerfeld Back-Propagator
import math as m
import matplotlib.image as mpimg
from scipy import signal
import numpy as np
import scipy
from scipy import ndimage


img = mpimg.imread('131118-1.png')
#img = mpimg.imread('MF1_30Hz_200us_away_median.png')

#%%
I0 = signal.medfilt2d(img, kernel_size = 3)
I0[I0 == 0] = np.median(I0) 

b = np.complex128(img/I0-1)

size = np.shape(b)
nx = size[0]
ny = size[1]

lambdaa = 0.525                                  # Wavelength of field
nm = 1                                           # Refraction index of medium
k = (2*m.pi/lambdaa)*nm                               # wavenumber 

x = np.arange(nx, dtype = np.float)
y = np.arange(ny, dtype = np.float)
z = np.arange(1,200, dtype = np.float)
xx,yy,zz = np.meshgrid(x,y,z)
del x,y

B = np.fft.fft2(b)
B = np.fft.fftshift(B)
#%%
#hZ = lambda x,y,z: (2*m.pi)**(-1)*z*(np.exp(1j*k*(x**2+y**2+z**2)**(1/2)))*\
#                    (1j*k*(x**2+y**2+z**2)**(-1)-(x**2+y**2+z**2)**(-3/2))
#hz = hZ(xx,yy,-zz)
#del xx,yy,zz
#Hz = np.fft.fft2(hz)
#
#Es = np.empty([nx,ny,0], dtype = np.complex)
##BH = np.empty_like(B)    
#BH = np.empty([nx,ny], dtype = np.complex)
#for ii in range(Hz.shape[2]):
#    BH = B*Hz[:,:,ii]
#    BH = np.exp(-1j*k*z[ii])*np.fft.ifft(np.fft.ifftshift(BH))
#    Es = np.dstack((Es,BH))
#del BH
      
#%%
#hZ = lambda x,y,z: (2*m.pi)**(-1)*z*(np.exp(1j*k*(x**2+y**2+z**2)**(1/2)))*\
#                    (1j*k*(x**2+y**2+z**2)**(-1)-(x**2+y**2+z**2)**(-3/2))
E = np.fft.fft2(b)
E = np.fft.fftshift(E)

qx = np.arange(0,nx)/nx-0.5
qy = np.arange(0,ny)/ny-0.5 
qx, qy = np.meshgrid(qx,qy)

qsq = ((lambdaa*np.sqrt((qx**2+qy**2)))/(2*m.pi*nm))
qfactor = np.sqrt(k*(1-qsq))    
ik = 1j*qfactor

Es = np.empty([ny,ny,z.shape[0]], dtype = np.complex) 
#Ess = np.empty([ny,ny,z.shape[0]]) 
   
for ii in range(z.shape[0]):
    Hqz = np.exp(ik*z[ii])
    TE = E*Hqz
    TE = np.fft.ifftshift(TE)
    TE = np.fft.fft2(TE)
    Es[:,:,ii] = TE
    Es[:,:,ii] = np.rot90(np.rot90(Es[:,:,ii]))
    dx = ndimage.sobel(abs(Es[:,:,ii]), 1)
    dy = ndimage.sobel(abs(Es[:,:,ii]), 0)
    Es[:,:,ii] = np.hypot(dx, dy)
    
    
#%% Sobel filter    
#import scipy
#from scipy import ndimage
#
#Es = abs(Es)
#dx = ndimage.sobel(Es[:,:,1], 1)  # horizontal derivative
#dy = ndimage.sobel(Es[:,:,1], 0)  # vertical derivative
#Ess = np.hypot(dx, dy)  # magnitude
##Ess *= 255.0 / np.max(Ess)  # normalize (Q&D)
#scipy.misc.imsave('sobel.jpg', Ess)
#%%
import matplotlib.pyplot as plt
plt.imshow(abs(Es[:,:,198]), cmap = 'gray')    
scipy.misc.imsave('198.png',abs(Es[:,:,198]))    
