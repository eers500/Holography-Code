# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

T0 = time.time()
CAP = cv2.VideoCapture('131118-1.avi')
NUM_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))

IMG = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), np.dtype('uint8'))
IM_STACK = np.empty((NUM_FRAMES, HEIGHT, WIDTH))

I = 0
SUCCESS = True

while (I < NUM_FRAMES  and SUCCESS):
    SUCCESS, IMG[I] = CAP.read()
    IM_STACK[I] = IMG[I, :, :, 1]
#    STACK[FC] = 
    I += 1
    print(I)

CAP.release()
IM_STACK = np.swapaxes(np.swapaxes(IM_STACK, 0, 2), 0, 1)
T = time.time()-T0
print(T)      
#%%
#for i in range(imStack.shape[2]):
#    plt.imshow(imStack[:,:,i])
#    plt.pause(0.01)