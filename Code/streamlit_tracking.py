#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:04:09 2022

@author: erick
"""

""" Rayleigh-Sommerfeld Propagator"""
#%%
import streamlit as st
import os
import math as m
import time
import matplotlib.image as mpimg
import numpy as np
import easygui
import matplotlib.pyplot as plt
from functions import bandpassFilter, exportAVI, histeq


I = mpimg.imread('/media/erick/NuevoVol/LINUX_LAP/PhD/Holography_Videos/0um_10x_ECOLI_HCB1_100hz_40us.png')
IB = mpimg.imread('/media/erick/NuevoVol/LINUX_LAP/PhD/Holography_Videos/MED_0um_10x_ECOLI_HCB1_100hz_40us.png')
IB[IB == 0] = np.mean(IB)
IN = I/IB   #divide

st.image(I)

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

IM = (IZ - np.min(IZ))*(255/(np.max(IZ)-np.min(IZ)))
IMM = np.uint8(IM)

kk = st.slider('frame', min_value=0, max_value=149, step=1)
st.image(IMM[:, :, kk])
