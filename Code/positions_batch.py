# -*- coding: utf-8 -*-
"""Calculate position of particles for several frames"""
#%% Import vido and set paramaters
import time
import numpy as np
import easygui as gui
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
from progress.bar import Bar


# PATH = gui.easygui.fileopenbox()
# PATH = 'MF1_30Hz_200us_awaysection.avi'
#PATH = '10x_laser_50Hz_10us_g1036_bl1602_500frames.avi'
# PATH = '/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames.avi'
PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/Pseudomonas/2017-10-23/red_laser_100fps_200x_0_135msec_1/red_laser_100fps_200x_0_135msec_1.avi'
# THRESHOLD = gui.enterbox(msg='Threshold for Gradient Stack', title='Threshold', default='0.1')

OPTIONS = gui.multenterbox(msg='Threshold and file export', title='Thresold and export',
                           fields=['THRESHOLD for GS:',
                                   'Number of frames for calculations:',
                                   'Export CSV file? (y/n):'])

T0 = time.time()
VID = f.videoImport(PATH, 0)
FRAMES_MEDIAN = 20
I_MEDIAN = f.medianImage(VID, FRAMES_MEDIAN)

N = 1.3226
LAMBDA = 0.642              # HeNe
MPP = 10
FS = 0.711                     # Sampling Frequency px/um
SZ = 10                       # # Step size um
NUMSTEPS = 150
THRESHOLD = np.float(OPTIONS[0])

#%% Test positions3D
# FRAME = 20
# I = VID[:, :, FRAME]
# IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, MPP, FS, SZ, NUMSTEPS)
# GS = f.zGradientStack(IM)  # GradientStack and RS propagator
# GS[GS < THRESHOLD] = 0
# LOCS = f.positions3D(GS)

# f.plot3D(LOCS, title='pos')

#%%  Calculate propagators, gradient stack and compute particle position ins 3D
if np.float(OPTIONS[1]) == -1:
    NUM_FRAMES = np.shape(VID)[-1]
else:
    NUM_FRAMES = int(OPTIONS[1])

# NUM_FRAMES = 5
VID = VID[:, :, :NUM_FRAMES]
LOCS = np.empty((NUM_FRAMES, 3), dtype=object)
# INTENSITY = np.empty(NUM_FRAMES, dtype=object)

# IMM = np.empty((512, 512, 150, NUM_FRAMES), dtype='float16')
# GSS = np.empty((512, 512, 150, NUM_FRAMES), dtype='float16')
T = []
T0 = time.time()
bar = Bar('Processing', max=NUM_FRAMES, suffix='%(percent).1f%% - %(eta)ds')

for i in range(NUM_FRAMES):
    I = VID[:, :, i]
    IM = f.rayleighSommerfeldPropagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS).astype('float32')
    # IMM[:, :, :, i] = IM
    GS = f.zGradientStack(IM).astype('float32')  
    GS[GS < THRESHOLD] = 0
    # GS = f.modified_propagator(I, I_MEDIAN, N, LAMBDA, FS, SZ, NUMSTEPS)  # Modified propagator
    # GSS[:, :, :, i] = GS
        
    # Turn to zero 99.99% of values
    # HIST, BINS = np.histogram(GSS.flatten(), 1000)
    # CDF = np.cumsum(HIST)
    # ID = np.where(CDF <= CDF.max()*0.9999)   
    # GS[GS <= BINS[ID[0][-1]]] = 0
    
    LOCS[i, 0] = f.positions3D(GS, peak_min_distance=20)
    A = LOCS[i, 0].astype('int')
    LOCS[i, 1] = IM[A[:, 0], A[:, 1], A[:, 2]]
    LOCS[i, 2] = GS[A[:, 0], A[:, 1], A[:, 2]]
    T.append(time.time()-T0)
    # print(str(i+1)+' of '+ str(NUM_FRAMES), (time.time()-T0))
    bar.next()
bar.finish()
print((time.time()-T0)/60)

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(len(T)), np.array(T)/60, '.-'); ax[0].grid()
ax[0].set_title('Computation time'); ax[0].set_xlabel('Number of frames'); ax[0].set_ylabel('Time (min)')

ax[1].plot(np.arange(1, len(T)), np.diff(T), '.-'); ax[1].grid()
ax[1].set_title('Computation time per frame'); ax[1].set_xlabel('Frame number'); ax[1].set_ylabel('Time (s)')
fig.show()

POSITIONS = pd.DataFrame(columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME'])
for i in range(np.shape(LOCS)[0]):
    XYZ, I_FS, I_GS, FRAME = LOCS[i, 0], LOCS[i, 1], LOCS[i, 2], i*np.ones_like(LOCS[i, 2])
    DATA = np.concatenate((XYZ, np.expand_dims(I_FS, axis=1), np.expand_dims(I_GS, axis=1), np.expand_dims(FRAME, axis=1)), axis=1)
    POSITIONS = POSITIONS.append(pd.DataFrame(DATA, columns=['X', 'Y', 'Z', 'I_FS', 'I_GS', 'FRAME']))

#%% Save to file    

if OPTIONS[2] == 'y':
    # POSITIONS.to_csv('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_2000frames_2000frames_modified_propagator_Results.csv', header=True)
    POSITIONS.to_csv(PATH[:-4]+'_'+np.str(NUM_FRAMES)+'_FRAMES_RS_TH'+np.str(THRESHOLD).replace('.','')+'.csv')  # For leptospira data
    print('Results exported to: \n', PATH[:-4]+'_'+np.str(NUM_FRAMES)+'_FRAMES_RS_TH'+np.str(THRESHOLD).replace('.','')+'.csv')

#%% Plot with f.plot3D
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot

# fig  = pyplot.figure()
# ax = Axes3D(fig)
# for i in range(NUM_FRAMES):
#     f.plot3D(LOCS[i, 0],'Cells in 3D', fig, ax); pyplot.show()
#     pyplot.pause(0.1)
    
#%% Plot with plotly.express
# import plotly.express as px
# import pandas as pd
# from plotly.offline import plot

# fig = px.scatter_3d(POSITIONS, x='X', y='Y', z='Z', color='FRAME')
# fig.update_traces(marker=dict(size=4))
# plot(fig)

#%% Plot with plotly.graph_objects
import plotly.graph_objects as go
from plotly.offline import plot

PATH = gui.fileopenbox(default='/home/erick/Documents/PhD/Colloids/')
POSITIONS = pd.read_csv(PATH, index_col=0)

fig = go.Figure(data=[go.Scatter3d(
    x=POSITIONS['X'], 
    y=POSITIONS['Y'], 
    z=POSITIONS['Z'],
    mode='markers', 
    marker=dict(
        size=1,
        color=POSITIONS['FRAME'].values,
        colorscale='Viridis'
        ),
    hovertext=['X+Y+Z+FRAME']
    
)])
fig.show()
plot(fig)

fig.write_html(PATH[:-3]+'html')
               
#%% Matplotlib scatter plot
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

# PKS = A.__array__()
PKS = POSITIONS.__array__()
# np.savetxt('locs.txt', PKS)
fig = pyplot.figure()
ax = Axes3D(fig)

# p = ax.scatter(PKS[:, 0], PKS[:, 1], PKS[:, 2], s=25, marker='o')
p = ax.scatter(POSITIONS['X'], POSITIONS['Y'], POSITIONS['Z'], s=2, marker='o', c=POSITIONS['FRAME'])

ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')
fig.colorbar(p)
pyplot.show()


