#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:54:08 2020

@author: erick
"""
import numpy as np
import pandas as pd 
import easygui as gui
import functions as f

PATH = gui.fileopenbox(msg='Select three files', title='File selection',
                       default='/media/erick/NuevoVol/LINUX_LAP/PhD', filetypes='*.csv', multiple=True)



# PATH1 = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv'
# PATH1 = '/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_500_FRAMES_RS_TH01.csv'

# PATH2 = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv'
# PATH2 = '/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_500_FRAMES_MODIFIED.csv'

P_RS = pd.read_csv(PATH[0])
P_MOD = pd.read_csv(PATH[1])

T = pd.read_csv(PATH[2])

#%% Plot with plotly.graph_objects
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
    subplot_titles=(PATH[0].split('/')[-1], PATH[1].split('/')[-1], PATH[2].split('/')[-1])
)

fig.add_trace(go.Scatter3d(x=P_RS['X'], y=P_RS['Y'], 
                           z=P_RS['Z'], mode="markers", 
                           marker=dict(
                               size=1,
                               color=P_RS['FRAME'].values,
                               colorscale='Viridis'
                               )),
              row=1, col=1)

fig.add_trace(go.Scatter3d(x=P_MOD['X'], y=P_MOD['Y'], 
                           z=P_MOD['Z'], mode="markers",
                           marker=dict(
                               size=1,
                               color=P_MOD['FRAME'].values,
                               colorscale='Viridis'
                               )),
              row=1, col=2)

fig.add_trace(go.Scatter3d(x=T['X'], y=T['Y'], 
                           z=T['Z'], mode="markers",
                           marker=dict(
                               size=1,
                               color=T['FRAME'],
                               colorscale='Viridis'
                               )),
              row=1, col=3)

fig.update_layout(height=900, width=1800, title_text="Propagator Comparison")
fig.show()
plot(fig)

#%%

import numpy as np
import pandas as pd 
import easygui as gui
import functions as f

PATH = gui.fileopenbox(msg='Select three files', title='File selection',
                       default='/media/erick/NuevoVol/LINUX_LAP/PhD', filetypes='*.csv', multiple=True)



# PATH1 = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv'
# PATH1 = '/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_500_FRAMES_RS_TH01.csv'

# PATH2 = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv'
# PATH2 = '/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_500_FRAMES_MODIFIED.csv'

P_RS = pd.read_csv(PATH[0])
P_MOD = pd.read_csv(PATH[1])

#%%

import numpy as np
import pandas as pd 
import easygui as gui
import functions as f

PATH = gui.fileopenbox(msg='Select three files', title='File selection',
                       default='/media/erick/NuevoVol/LINUX_LAP/PhD', filetypes='*.csv', multiple=True)



# PATH1 = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv'
# PATH1 = '/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_500_FRAMES_RS_TH01.csv'

# PATH2 = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv'
# PATH2 = '/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1_500_FRAMES_MODIFIED.csv'

P_RS = pd.read_csv(PATH[0])
P_MOD = pd.read_csv(PATH[1])

#% Plot with plotly.graph_objects
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    subplot_titles=(PATH[0].split('/')[-1], PATH[1].split('/')[-1])
)

fig.add_trace(go.Scatter3d(x=P_RS['X'], y=P_RS['Y'], 
                           z=P_RS['Z'], mode="markers", 
                           marker=dict(
                               size=1,
                               color=P_RS['FRAME'].values,
                               colorscale='Viridis'
                               )),
              row=1, col=1)

fig.add_trace(go.Scatter3d(x=P_MOD['X'], y=P_MOD['Y'], 
                           z=P_MOD['Z'], mode="markers",
                           marker=dict(
                               size=1,
                               color=P_MOD['FRAME'].values,
                               colorscale='Viridis'
                               )),
              row=1, col=2)



fig.update_layout(height=900, width=1800, title_text="Hungrarian Algorithm")
fig.show()
plot(fig)

# fig.write_html('/home/erick/HA_Res.html')


# %%
