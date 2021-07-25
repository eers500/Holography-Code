#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:02:28 2021

@author: erick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import easygui as gui
import os

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/')
export_path = os.path.split(path)
export_path = export_path[:-1][0]+'/frame_files'
data = pd.read_csv(path, index_col=0)

#%%
frame_number = data['FRAME']
fn = np.uint(frame_number.unique())
ending = pd.DataFrame(-np.ones((1, 4)), columns=['I_GS', 'X', 'Y', 'Z'])

for k in range(len(fn)):
    frame_data = data[data['FRAME'] == fn[k]]
    frame_data = frame_data[['I_GS', 'X', 'Y', 'Z']]
    frame_data = frame_data.append(ending, ignore_index=True)
    frame_data.to_csv(export_path+'/frame{0}.txt'.format(k), sep=' ', header=False, index=False)
    print(k)
    