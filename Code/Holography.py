# -*- coding: utf-8 -*-
import matplotlib.image as mpimg
import numpy as np
# Read image
import matplotlib.pyplot as plt
from functions import Bandpass_Filter

img = mpimg.imread('MF1_30Hz_200us_away_median.png')

# Large cutoff size (Pixels)
xl = 50
# Small cutoff size (Pixels)
xs = 20

hol = Bandpass_Filter(img,xl,xs)

plt.imshow(hol,cmap='gray')