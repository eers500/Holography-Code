# -*- coding: utf-8 -*-
from functions import Bandpass_Filter,videoImport
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
#%% Extract a frame and apply Bandpass. Repeat for al frames
t0 = time.time()
vid = "MF1_30Hz_200us_awaysection.avi"
vidcap = cv2.VideoCapture(vid)
num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success,image = vidcap.read()
frame1 = image[:,:,0]
BP_array = np.empty_like(frame1)

xl = 120
xs = 50

t0 = time.time()
for ii in range(1,num_frames):
        print(ii)
        success,fr = vidcap.read()
        frame = fr[:,:,0]
        BP = Bandpass_Filter(frame,xl,xs)
        BP_array = np.dstack((BP_array,BP))
time1 = time.time()-t0       
#plt.figure(1)
#plt.imshow(frame)               16 minutes                
#
#plt.figure(2)
#plt.imshow(BP_array)        
#%% Extract all frames in 3D array and apply Bandpass filter to every slice
t1 = time.time()
vid = "MF1_30Hz_200us_awaysection.avi"
imStack = videoImport(vid)
xl = 120
xs = 50

t0 = time.time()
BP_array = np.empty([imStack.shape[0],imStack.shape[1],1])
for ii in range(1,imStack.shape[2]):
    print(ii)
    BP = Bandpass_Filter(imStack[:,:,ii],xs,xl)
    BP_array = np.dstack((BP_array,BP))
time = time.time()-t1    
    
#  16.5
#%%
import av
import av.datasets

vid = "MF1_30Hz_200us_awaysection.avi"
container = av.open(vid)

# Signal that we only want to look at keyframes.
stream = container.streams.video[0]
stream.codec_context.skip_frame = 'NONKEY'

for frame in container.decode(stream):

    print(frame)

    # We use `frame.pts` as `frame.index` won't make must sense with the `skip_frame`.
    frame.to_image().save(
        'night-sky.{:04d}.jpg'.format(frame.pts),
        quality=80,
    )
#%%
import av    
for frame in container.decode(video=0):
    frame.to_image.

