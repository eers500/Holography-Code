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
FS = 0.711                     # Sampling Frequency px/um
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
Q = np.empty_like(IN, dtype='complex64')
for i in range(NI):
    for j in range(NJ):
        Q[i, j] = ((LAMBDA*FS)/(max([NI, NJ])*N))**2*((i-NI/2)**2+(j-NJ/2)**2)
# P = np.conj(P)
Q = np.sqrt(1-Q)-1

if all(Z > 0):
    Q = np.conj(Q)

# R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
IZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')

T0 = time.time()
for k in range(Z.shape[0]):
    R = np.exp((-1j*K*Z[k]*Q), dtype='complex64')
    IZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E*R)))

print(time.time() - T0)

#%%
# scipy fft
#import scipy.fftpack as ft
## R = np.empty([NI, NJ, Z.shape[0]], dtype=complex)
#IZZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')
#
#T0 = time.time()
#for k in range(Z.shape[0]):
#    R = np.exp((-1j*K*Z[k]*Q), dtype='complex64')
#    IZZ[:, :, k] = np.real(1 + ft.ifft2(ft.ifftshift(E*R)))
#
#print(time.time() - T0)

#%%
# plt.imshow(IZ[:, :, 149], cmap = 'gray')

#%%
IM = (IZ - np.min(IZ))*(255/(np.max(IZ)-np.min(IZ)))
IMM = np.uint8(IM)
# EX_PATH, NAME = os.path.split(PATH[0])
# exportAVI(EX_PATH+NAME[0:-4]+'_frame_stack_'+str(SZ)+'um.avi', IMM, IMM.shape[0], IMM.shape[1], 30)

EX_PATH, NAME = os.path.split(PATH)
exportAVI(EX_PATH+'/'+NAME[0:-4]+'_frame_stack_'+str(SZ)+'um.avi', IMM, IMM.shape[0], IMM.shape[1], 30)

#%%
# Histogram equalizaion for visualization
# IZZ, CDF = histeq(IM)
# IZZZ = np.uint8(IZZ)
# exportAVI('IZZZ.avi',IZZZ, IZZ.shape[0], IZZ.shape[1], 30)

#%%
# Symbolic gradient stack
import numpy as np
from sympy import *

x, y, z, k = symbols('x y z k')
e = exp(1j*k*(x**2+y**2+z**2)**(1/2))/(x**2+y**2+z**2)**(1/2)
h = (1/2*np.pi)*diff(e, z)
H = diff(h, z)

hh = lambdify((x, y, z, k), h, 'numpy')
HH = lambdify((x, y, z, k), H, 'numpy')

#%%
X = np.arange(np.shape(IN)[0])-np.shape(IN)[0]/2
Y = X

xx, yy = np.meshgrid(X, Y)

rs = hh(xx, yy, Z[149], K)
GS = HH(xx, yy, Z[149], K)

# plt.figure(1)
# plt.imshow(np.real(gs), cmap='gray')
# plt.show()ddd
# plt.figure(2)
# plt.imshow(np.real(GS), cmap='gray')
# plt.show()

IZZ = np.empty([NI, NJ, Z.shape[0]], dtype='float32')
R = np.empty([NI, NJ, Z.shape[0]], dtype='complex64')
for k in range(1,Z.shape[0]):
    print(k)
    R[:, :, k] = np.fft.fftshift(np.fft.fft2(HH(xx, yy, Z[k], K)))
    IZZ[:, :, k] = np.real(1 + np.fft.ifft2(np.fft.ifftshift(E*R[:, :, k])))