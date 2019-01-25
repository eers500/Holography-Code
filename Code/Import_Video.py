# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

vidcap = cv2.VideoCapture("MF1_30Hz_200us_awaysection.avi")
success,image = vidcap.read()
num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
imStack = image[:,:,0]
i = 1
#imStack.reshape(())
#while success:    
#    success,fr = vidcap.read()
#    frame = fr[:,:,0]
#    imStack = np.dstack((imStack,frame))
#    i = i+1

t0 = time.time()
for ii in range(1,num_frames):
        success,fr = vidcap.read()
        frame = fr[:,:,0]
        imStack = np.dstack((imStack,frame))
t1 = time.time()
time = t1-t0        
#%%
for i in range(imStack.shape[2]):
    plt.imshow(imStack[:,:,i])
    plt.pause(0.01)