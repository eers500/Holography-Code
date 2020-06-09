#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:07:13 2020

@author: erick
"""


# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('figure',  figsize=(10, 6))
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

#%
frames = np.load('/home/erick/Documents/PhD/Colloids/Colloids_4D_array_IM_100frames.npy')

#%%
frames = np.swapaxes(frames, 0, 3)
frames = np.swapaxes(frames, 1, 3)
frames = np.swapaxes(frames, 2, 3)

#%%
for i in range(len(frames)):
    print(i)
    frames[i, :, :, :] = 255 * frames[i, :, :, :]  / frames[i, :, :, :].max()
    
frames = frames.astype('uint8')

#%%
# features = tp.locate(frames[:, :, :, 0].astype('float32'), diameter=(25, 25, 25))


# tp.annotate3d(features, frames[:, :, :, 0])



f = tp.batch(frames, diameter=(35, 35, 35), separation=(35, 35, 35))

#%%



f['xum'] = f['x'] * 0.21
f['yum'] = f['y'] * 0.21
f['zum'] = f['z'] * 0.75

    

for search_range in [1.0, 1.5, 2.0, 2.5]:
    linked = tp.link_df(f, search_range, pos_columns=['xum', 'yum', 'zum'])
    hist, bins = np.histogram(np.bincount(linked.particle.astype(int)),
                              bins=np.arange(30), normed=True)
    plt.step(bins[1:], hist, label='range = {} microns'.format(search_range))
plt.ylabel('relative frequency')
plt.xlabel('track length (frames)')
plt.legend();


#%%
#% Matplotlib scatter plot
# 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot

# # PKS = A.__array__()
# # np.savetxt('locs.txt', PKS)
# fig = pyplot.figure()
# ax = Axes3D(fig)

# # p = ax.scatter(PKS[:, 0], PKS[:, 1], PKS[:, 2], s=25, marker='o')
# p = ax.scatter(linked['z'], linked['y'], linked['x'], s=5, marker='o', c=linked['particle'])

# # ax.tick_params(axis='both', labelsize=10)
# # ax.set_title('Cells Positions in 3D', fontsize='20')
# # ax.set_xlabel('x (pixels)', fontsize='18')
# # ax.set_ylabel('y (pixels)', fontsize='18')
# # ax.set_zlabel('z (slices)', fontsize='18')
# fig.colorbar(p)
# pyplot.show()


#%% Plot with plotly.graph_objects
# import plotly.graph_objects as go
# from plotly.offline import plot

# fig = go.Figure(data=[go.Scatter3d(
#     x=linked['z'], 
#     y=linked['y'], 
#     z=linked['x'],
#     mode='markers', 
#     marker=dict(
#         size=1,
#         color=linked['particle'].values,
#         colorscale='Viridis'
#         ),
#     hovertext=['X+Y+Z+FRAME']
    
# )])
# fig.show()
# plot(fig)