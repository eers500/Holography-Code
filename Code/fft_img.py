#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Read image and FFT it
import matplotlib.image as mpimg
import numpy as np
# Read image
import matplotlib.pyplot as plt

img = mpimg.imread('sphere.jpg')

# Convert image to grayscale
from functions import rgb2gray
img_gs = rgb2gray(img)
#%% Show graysacle image
from matplotlib.pyplot import imshow
plt.subplot(221)
plt.imshow(img_gs, cmap="gray")
#%% FFT the grayscale image
imgfft = np.fft.fft2(img_gs)
img_fft = np.fft.fftshift(imgfft)
img_amp = abs(img_fft)
del imgfft

plt.subplot(222)
plt.imshow(np.log10(img_amp),cmap="gray")
plt.show()

#%% Distance form zero frequency
fzero_ids = np.where(img_amp==np.max(img_amp))
fzero_id = [int(fzero_ids[0]),int(fzero_ids[1])]


[nx,ny]=img_amp.shape

r = np.empty([nx,ny])
for ii in range(0,nx-1):
    for jj in range(0,ny-1):
        r[ii, jj] = np.sqrt((fzero_id[0]-ii)**2+(fzero_id[1]-jj)**2)


#%% Create bandpass filter
cutoffMax = 50
cutoffMin = 30

filter1 = np.empty([nx-1,ny-1])
filter2 = np.empty([nx-1,ny-1])
filter3 = np.empty([nx-1,ny-1])
for ii in range(0,nx-1):
    for jj in range(0,ny-1):
        filter1[ii, jj] = np.exp(-r[ii, jj]**2/(2*cutoffMax**2))
        filter2[ii, jj] = -np.exp(-r[ii, jj]**2/(2*cutoffMin**2))
        filter3[ii, jj] = filter1[ii, jj] + filter2[ii, jj]
        
plt.subplot (223)
plt.imshow(filter3,cmap='gray')
        
#%% Update image
filtered = filter3*img_fft[0:-1,0:-1]
img_filt = np.fft.ifftshift(filtered) 
img_filt= np.fft.fft2(img_filt)       
img_filt = np.rot90(np.real(img_filt),2)

plt.subplot(224)
plt.imshow(img_filt,cmap='gray')
plt.show()
        
