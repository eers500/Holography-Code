#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:07:12 2020

@author: erick
"""

from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt

# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

#%%
# FRAMES = pims.Video('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_BG_grayscale.avi')
frames = pims.TiffStack('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_BG_grayscale.tif')

f = tp.locate(frames[0], 29, invert=True)

plt.figure()  # make a new figure
tp.annotate(f, frames[0]);

f = tp.batch(frames[:], 29, minmass=200, invert=True);


t = tp.link_df(f, 5, memory=3)


t1 = tp.filter_stubs(t, 10)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())



plt.figure()
tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass


# t2 = t1[((t1['mass'] > 250) & (t1['size'] < 3.0) &
#          (t1['ecc'] < 0.1))]

t2 = t1

plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0]);



plt.figure()
tp.plot_traj(t1);

