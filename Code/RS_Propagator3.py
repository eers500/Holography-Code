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
#%% Median image
MI