#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:54:08 2020

@author: erick
"""
import numpy as np
import pandas as pd 
import functions as f

P_RS = pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_rayleighSommerfeld_Results.csv')
P_MOD = pd.read_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv')

#%% Plot with plotly.graph_objects
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    subplot_titles=("Rayleigh-Sommerfeld", "Modified")
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

fig.update_layout(height=900, width=1800, title_text="Propagator Comparison")
fig.show()
plot(fig)
