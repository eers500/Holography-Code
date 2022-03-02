# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 00:53:51 2021

@author: eers500
"""

import numpy as np
import pandas as pd
import functions as f
import sklearn.cluster as cl
import easygui as gui
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


path = gui.fileopenbox()
DF = pd.read_csv(path)

#%%
cores = os.cpu_count()

eps = 20 #10
min_samples = 10

DBSCAN = cl.DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=cores).fit(DF[['X', 'Y', 'Z', 'FRAME']])
LINKED = DF.copy()
LINKED['PARTICLE'] = DBSCAN.labels_
LINKED = LINKED.drop(np.where(LINKED.PARTICLE.values == -1)[0])

#%% Track with search sphere

rsphere = 20
frame_skip = 5
min_size = 50

# dd = DF[DF['FRAME'] == 0]
# frames = np.unique(DF['FRAME'])

dd = S
frames = np.unique(dd['FRAME'])

tracks = []
for n in tqdm(range(len(dd))):

    # n = 70
    x0, y0, z0, t0, fr0 = dd['X'][n], dd['Y'][n], dd['Z'][n], dd['TIME'][n], dd['FRAME'][n]
    
    track = [(x0, y0, z0, t0, fr0, n)]
    frame_skip_counter = 0
    
    for k in range(1, len(frames)):
        s = DF[DF['FRAME'] == k]
        x, y, z, tt, fr = s['X'].values, s['Y'].values, s['Z'].values, s['TIME'].values, s['FRAME'].values
        
        dist = np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)
        
        if (dist<rsphere).any():
            id_min = np.where(dist == dist.min())[0][0]
            track.append((x[id_min], y[id_min], z[id_min], tt[id_min], fr[id_min], n))
            x0, y0, z0 = x[id_min], y[id_min], z[id_min]
            frame_skip_counter = 0
        else:
            frame_skip_counter += 1
            if frame_skip_counter > frame_skip:
                break
    
    if len(track) > min_size:
        track = pd.DataFrame(np.array(track), columns=['X', 'Y', 'Z', 'TIME', 'FRAME', 'PARTICLE'])
        tracks.append(track)
        

LINKED = pd.concat(tracks)    
#

# for t in tracks:
#     plt.plot(t[:, 0], -t[:, 1])
    
# plt.scatter(tracks[:, 0], -tracks[:, 1], s=1)




#%% 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

X = LINKED.X
Y = LINKED.Y
Z = LINKED.Z
T = LINKED.TIME
P = LINKED.PARTICLE

ax.scatter(Y, X, Z, s=2, marker='o', c=T)
pyplot.show()

#%% Plotly scatter plot
import plotly.express as px
import pandas as pd
import easygui as gui
from plotly.offline import plot

# fig = px.scatter_3d(LINKED, x='Y', y='X', z='Z', color='PARTICLE')
fig = px.line_3d(LINKED, x='Y', y='X', z='Z', color='PARTICLE')
# fig = px.line_3d(smoothed_curves_df, x='X', y='Y', z='Z', color='PARTICLE', hover_data=['TIME'])
fig.update_traces(marker=dict(size=1))

#fig.add_trace(fig2)
plot(fig)