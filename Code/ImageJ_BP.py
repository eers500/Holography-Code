# -*- coding: utf-8 -*-
# Read image and FFT it
import matplotlib.image as mpimg
import numpy as np
# Read image
import matplotlib.pyplot as plt
from functions import rgb2gray, square_image

img = mpimg.imread('aaa.png')
size = len(img.shape)
#[ni,nj] = img.shape

if size == 3:
    img_gs = rgb2gray(img)
    [ni,nj,nk] = img.shape
elif size == 2:
    [ni,nj] = img.shape    

if ni != nj:
    [img_gs,axis,dn] = square_image(img_gs)
else:
    img_gs = img
    axis = 'square'

del ni,nj
#plt.figure(1)
#plt.subplot(221)
plt.title('Sample Image')
plt.imshow(img_gs,cmap='gray')

#%% FFT the grayscale image
imgfft = np.fft.fft2(img_gs)
img_fft = np.fft.fftshift(imgfft)
img_amp = abs(img_fft)
del imgfft

#plt.figure(2)
#plt.subplot(222)
plt.title('Fourier Transform')
plt.imshow(np.log10(img_amp),cmap='gray')

#%% Pre filter image information
[ni,nj]=img_amp.shape
MIS = ni
 
# Large cutoff size (Pixels)
xl = 50
# Small cutoff size (Pixels)
xs = 20

#%% Create bandpass filter when BigAxis == 
LCO = np.empty([ni,nj])
SCO = np.empty([ni,nj])

for ii in range(0,ni-1):
    for jj in range(0,nj-1):
        LCO[ii, jj] = np.exp(-((ii-MIS/2)**2+(jj-MIS/2)**2)*(2*xl/MIS)**2)
        SCO[ii, jj] = np.exp(-((ii-MIS/2)**2+(jj-MIS/2)**2)*(2*xs/MIS)**2)
        
BP = SCO - LCO

#plt.figure(3)
#plt.subplot(223)
plt.title('Bandpass Filter')
plt.imshow(BP,cmap='gray')        
     
#%% Filter image 
filtered = BP*img_fft
img_filt = np.fft.ifftshift(filtered) 
img_filt= np.fft.fft2(img_filt)       
img_filt = np.rot90(np.real(img_filt),2)

if axis == 'i':
    img_filt = img_filt[0:-dn,:]
elif axis == 'j':
    img_filt = img_filt[:,0:-dn]
    

#plt.figure(4)
#plt.subplot(224)
plt.title('Filtered Image')
plt.imshow(img_filt, cmap = 'gray')
plt.show()

#%%
#import scipy.misc
#scipy.misc.imsave('img_filt.png', img_filt)
#scipy.misc.imsave('img_gs.png', img_gss)                
#%%
#imgJ = mpimg.imread('img_filt_imJ.png')
#diff = img

       