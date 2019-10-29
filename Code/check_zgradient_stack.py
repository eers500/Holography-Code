#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:48:34 2019

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f


IM = f.videoImport('E:/PhD/23_10_19/000_L_10x_100Hz_45us_frame_stack_10um.avi', 0)
GS = f.zGradientStack(IM)
grad = -np.gradient(IM, axis=-1)

#%%
GS[GS < 100] = 0
GSS = np.uint8((GS/np.max(GS))*255)

grad[grad < 5] = 0
gradd = np.uint8((grad/np.max(grad))*255)

#%%
plt.figure(1)
plt.imshow(GS[:, :, 35], cmap='gray')
plt.show()

plt.figure(2)
plt.imshow(grad[:, :, 35], cmap='gray')
plt.show()

#%%}
#f.imshow_sequence(GS, 0.1, True)
#f.imshow_sequence(grad, 0.1, True)

#%%
# 334, 
plt.imshow(np.rot90(IM[188, :, :]), cmap='gray')
plt.show()

#%%
# 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
#
#
# fig = pyplot.figure()
# ax = Axes3D(fig)
#
# #ni, nj, _ = np.shape(IM)
# xx = np.arange(150)
# yy = np.arange(510)
#
# X, Y = np.meshgrid(xx,yy)
#
# p = ax.plot_surface(X, Y, IM[334, :, :])
# ax.tick_params(axis='both', labelsize=10)
# #ax.set_title('Cells Positions in 3D', fontsize='20')
# #ax.set_xlabel('x (pixels)', fontsize='18')
# #ax.set_ylabel('y (pixels)', fontsize='18')
# #ax.set_zlabel('z (slices)', fontsize='18')
# fig.colorbar(p)
# pyplot.show()

#%%
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

fig = go.Figure(data=[go.Surface(z=IM[334, :, :])])
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.show()
plot(fig)
