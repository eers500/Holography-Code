# -*- coding: utf-8 -*-
""" Rayleigh-Sommerfeld Propagator"""
#%%
import os
import math as m
import time
import matplotlib.image as mpimg
import numpy as np
import easygui
from functions import bandpassFilter, exportAVI, histeq

T0 = time.time()
FILES = easygui.fileopenbox(multiple=True)

PATH = [[],[]]
if FILES[0].find('MED') != -1:
    PATH[0] = FILES[1]
    PATH[1] = FILES[0]
else:
    PATH = FILES

# PATH_I = 'FRAMES/10um_10x_ECOLI_HCB1_100hz_40us.png'
# I = mpimg.imread('131118-1.png')
# I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
# I = mpimg.imread('10x_laser_50Hz_10us_g1036_bl1602-003.png')
# I = mpimg.imread('23-09-19_ECOLI_HCB1_100Hz_50us_10x_3-frame(0).png')
I = mpimg.imread(PATH[0])

#%%
# Median image
# PATH_IB = easygui.fileopenbox()
# PATH_IB = 'FRAMES/MED_0um_10x_ECOLI_HCB1_100hz_40us-1.png'

# IB = mpimg.imread('AVG_131118-1.png')
# IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')
# IB = mpimg.imread('MED_10x_laser_50Hz_10us_g1036_bl1602-003-1.png')
# IB = mpimg.imread('MED_23-09-19_ECOLI_HCB1_100Hz_50us_10x_3-1.png')
IB = mpimg.imread(PATH[1])


IB[IB == 0] = np.mean(IB)
IN = I/IB   #divide
# PATH = '/home/erick/Documents/PhD/23_10_19/300_L_10x_100Hz_45us.tif'
# IN = mpimg.imread(PATH)


N = 1.3226
LAMBDA = 0.642               # Diode
#MPP = 20                      # Magnification: 10x, 20x, 50x, etc
2                     # Sampling Frequency px/um
NI = np.shape(IN)[0]
NJ = np.shape(IN)[1]
SZ = 10                       # Step size in um
# Z = (FS*(51/31))*np.arange(0, 150)       # Number of steps
Z = SZ*np.arange(0, 150)
# ZZ = np.linspace(0, SZ*149, 150)
# Z = FS*ZZ
K = 2*m.pi*N/LAMBDA            # Wavenumber

_, BP = bandpassFilter(IN, 2, 30)
E = np.fft.fftshift(BP)*np.fft.fftshift(np.fft.fft2(IN - 1))

#%%
X = np.arange(0,NI)
Y = np.arange(0,NJ)

# -*- coding: utf-8 -*-
""" Rayleigh-Sommerfeld Propagator"""
#%%
import os
import math as m
import time
import matplotlib.image as mpimg
import numpy as np
import easygui
from functions import bandpassFilter, exportAVI, histeq

T0 = time.time()
FILES = easygui.fileopenbox(multiple=True)

PATH = [[],[]]
if FILES[0].find('MED') != -1:
    PATH[0] = FILES[1]
    PATH[1] = FILES[0]
else:
    PATH = FILES

# PATH_I = 'FRAMES/10um_10x_ECOLI_HCB1_100hz_40us.png'
# I = mpimg.imread('131118-1.png')
# I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
# I = mpimg.imread('10x_laser_50Hz_10us_g1036_bl1602-003.png')
# I = mpimg.imread('23-09-19_ECOLI_HCB1_100Hz_50us_10x_3-frame(0).png')
I = mpimg.imread(PATH[0])

#%%
# Median image
# PATH_IB = easygui.fileopenbox()
# PATH_IB = 'FRAMES/MED_0um_10x_ECOLI_HCB1_100hz_40us-1.png'

# IB = mpimg.imread('AVG_131118-1.png')
# IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')
# IB = mpimg.imread('MED_10x_laser_50Hz_10us_g1036_bl1602-003-1.png')
# IB = mpimg.imread('MED_23-09-19_ECOLI_HCB1_100Hz_50us_10x_3-1.png')
IB = mpimg.imread(PATH[1])


IB[IB == 0] = np.mean(IB)
IN = I/IB   #divide
# PATH = '/home/erick/Documents/PhD/23_10_19/300_L_10x_100Hz_45us.tif'
# IN = mpimg.imread(PATH)


N = 1.3226
LAMBDA = 0.642               # Diode
#MPP = 20                      # Magnification: 10x, 20x, 50x, etc
2                     # Sampling Frequency px/um
NI = np.shape(IN)[0]
NJ = np.shape(IN)[1]
SZ = 10                       # Step size in um
# Z = (FS*(51/31))*np.arange(0, 150)       # Number of steps
Z = SZ*np.arange(0, 150)
# ZZ = np.linspace(0, SZ*149, 150)
# Z = FS*ZZ
K = 2*m.pi*N/LAMBDA            # Wavenumber

_, BP = bandpassFilter(IN, 2, 30)
E = np.fft.fftshift(BP)*np.fft.fftshift(np.fft.fft2(IN - 1))

#%%
X = np.arange(-NI/2,NI/2)
Y = np.arange(-NJ/2,NJ/2)
X, Y = np.meshgrid(X,Y)

R2 = X**2+Y**2

# U = np.empty([NI, NJ, len(Z)], dtype='complex64')
# for k in range(len(Z)):
#     for i in range(len(X)):
#         for j in range(len(Y)):
#             print(i,j,k)
#             r = R2+ Z[k]
#             U[i, j, k] = -R2[i, j]*K*Z[k]**2

U = np.empty([NI, NJ, len(Z)], dtype='complex64')
H = U
for k in range(len(Z)):
    ZZ = np.ones_like(R2)*Z[k]
    U[:, :, k] = -R2*K**2*ZZ**2 - ZZ**4*K**2-R2**2+ZZ*3 + 1j*K*np.sqrt(R2+ZZ**2)*(R2-2*ZZ**2)
    H[:, :, k] = 1/(2*m.pi) * np.exp(1j*K*np.sqrt(R2+ZZ**2)) / (R2+ZZ**2)^(5/2)