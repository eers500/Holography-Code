# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:27:39 2019

@author: eers500
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('MF1_30Hz_200us_awaysection.avi')
cap.isOpened()

count = 0
succes = True
while success:
    succes,image = cap.read()
#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2

vidcap = cv2.VideoCapture("MF1_30Hz_200us_awaysection.avi")
image = vidcap.read()
num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))



#print(vidcap.read())
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1