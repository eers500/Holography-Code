#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 21:08:38 2022

@author: erick
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import easygui as gui
import functions as f
from tqdm import tqdm
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

path = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/Results/1/')
smoothed_curves_df = pd.read_csv(path)

#%%

particle_num = np.unique(smoothed_curves_df['PARTICLE'])

xx, yy, zz, tt, pp, sp = -1, -1, -1, -1, -1, -1

for pn in particle_num:
    s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    # print(pn, len(s))

    if len(s) > 100:
        speed, x, y, z, t = f.get_speed(s)
        xx = np.hstack((xx, x))
        yy = np.hstack((yy, y))
        zz = np.hstack((zz, z))
        tt = np.hstack((tt, t))
        pp = np.hstack((pp, pn*np.ones(len(t))))
        sp = np.hstack((sp, speed))
    
# expath = '/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/Results/1/Archea/Plots/'
tracks_w_speed = pd.DataFrame(np.transpose([xx[1:], yy[1:], zz[1:], tt[1:], pp[1:], sp[1:]]), columns=['X', 'Y', 'Z', 'TIME', 'PARTICLE', 'SPEED'])
tracks_w_speed.to_csv(path[:-4]+'_speed.csv')


# PATH = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/', filetypes='.csv')
# tracks_w_speed = pd.read_csv(PATH, index_col=False)

fig = plt.figure(4, dpi=150)
ax = fig.add_subplot(111, projection='3d')

# p = ax.scatter(tracks_w_speed['Y'], tracks_w_speed['X'], tracks_w_speed['Z'], c=tracks_w_speed['SPEED'], marker='.', s=20)
# cbar = plt.colorbar(p)
# cbar.set_label('Speed ($\mu ms^{-1}$)')

for pn in particle_num:
    # s = smoothed_curves_df[smoothed_curves_df['PARTICLE'] == pn]
    s = tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
    ax.plot(s['X'], s['Y'], s['Z'], linewidth=2)
    # ax.scatter(s['Y'], s['X'], s['Z'])

ax.axis('tight')
ax.set_title('$\it{Escherichia \ Coli}$', fontsize=40)  # $\it{Escherichia \ Coli}$
ax.set_xlabel('y ($\mu$m)', fontsize=20)
ax.set_ylabel('x ($\mu$m)', fontsize=20)
ax.set_zlabel('-z ($\mu$m)', fontsize=20)
# ax.set_zlim(bottom=0, top=40)

plt.figure(3)
plt.hist(tracks_w_speed['SPEED'], 100)
mean_speed = tracks_w_speed['SPEED'].mean()
print(mean_speed)
plt.title('Speed: $\mu$ = ' + str(np.float16(mean_speed)) + ' $\mu m s^{-1}$', fontsize=40)
plt.xlabel('Speed ($\mu m s^{-1}$)', fontsize=20)
plt.ylabel('Frequency', fontsize=20)

pyplot.show()

#%%
import plotly.express as px
import pandas as pd
import easygui as gui
from plotly.offline import plot

pns = tracks_w_speed.PARTICLE.unique()
curve = tracks_w_speed[tracks_w_speed['PARTICLE'] == pns[1]]

# fig = px.line_3d(tracks_w_speed, x='X', y='Y', z='Z', color='PARTICLE', hover_data=['TIME'])
fig = px.scatter_3d(tracks_w_speed, x='X', y='Y', z='Z', color='SPEED')
fig.update_traces(marker=dict(size=1))

plot(fig)

#%% Re orientation Angle with RDP

def get_angles_rdp(sd, epsilon, degrees):
    """ Re-orientation event angle"""
    
    from rdp import rdp
    
    # ss = sd.values
    # x = ss[:, :3]
    # t = ss[:, 3]
    # p = ss[0, 4]
    
    x = sd[['X', 'Y', 'Z']].values
    t = sd['TIME'].values
    p = sd['PARTICLE'].values[0]
    
    xyz = rdp(x, epsilon)   # RDP reconstruction
    xyz = np.hstack((xyz, p*np.ones((len(xyz), 1))))
    
    
    cos = np.ones(len(xyz))
    Dr = np.ones(len(xyz))
    for i in range(1, len(xyz)-1):
        a1 = xyz[i-1, :]
        a2 = xyz[i, :]
        a3 = xyz[i+1, :]
        d21 = a2 - a1
        d32 = a3 - a2
        cos[i] = np.dot(d21, d32) / (np.sqrt(np.sum(d21**2)) * np.sqrt(np.sum(d32**2)))
        
    if degrees:
        angles = np.arccos(cos) * 180 / np.pi
        
    else:
        angles = np.arccos(cos)
    
    # plt.figure()
    # plt.plot(angle)
    
    # plt.figure()
    # plt.hist(angle, 30)
    
    # plt.figure(figsize=(9, 4.5))
    # ax0 = plt.subplot2grid((6, 6), (0, 0), 6, 3, projection='3d')
    # ax0.set_facecolor('none')
    # ax0.plot(sd.X, sd.Y, sd.Z, 'b-', label='original data')
    # # ax0.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'r.', label='Turns')
    # ax0.set_title('Smoothed track')

    # ax1 = plt.subplot2grid((6, 6), (0, 3), 6, 3, projection='3d')
    # ax1.set_facecolor('none')
    # ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'b-', label='RDP reconstructed eps = '+np.str(eps3d))
    # ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'r.', label='Turns')
    # ax1.set_title('RDP Simplification')
    # ax1.legend(loc='best')
    
    angles = np.hstack((angles.reshape((len(angles), 1)), p*np.ones((len(angles), 1))))
    
    return angles[1:-1], xyz

eps = 0.6
pnum = tracks_w_speed['PARTICLE'].unique()

angles = []
recs = []
for pn in tqdm(pnum):
    sd = tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
    a, rec = get_angles_rdp(sd, eps, True)
    angles.append(a)
    recs.append(rec)
    
angles = np.concatenate(angles)
recs = np.concatenate(recs)

angles = pd.DataFrame(angles, columns=['ANGLES', 'PARTICLE'])
recs= pd.DataFrame(recs, columns=['X', 'Y', 'Z', 'PARTICLE'])
amean = angles.ANGLES.mean()

#%%
plt.hist(angles['ANGLES'], 30)
plt.vlines(amean, ymin=0, ymax=50, colors='red')
plt.title('Mean: '+ str(round(amean, 3)))
plt.grid()

#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot


fig = pyplot.figure()
ax = Axes3D(fig)

pn = 9

sd =  tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
rec = recs[recs['PARTICLE'] == pn]

ax.plot(sd.X, sd.Y, sd.Z, linewidth=3)
ax.scatter(rec.X, rec.Y, rec.Z, s=50, marker='o', c='red')


ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (um)', fontsize='18')
ax.set_ylabel('y (um)', fontsize='18')
ax.set_zlabel('z (um)', fontsize='18')
# fig.colorbar(p, ax=ax)
pyplot.show()



#%% With LW metric
def xi_metric(sd):
    x = sd[['X', 'Y', 'Z']].values
    v = sd['SPEED'].values
    t = sd['TIME'].values
    
    
    vmean = v.mean()
    dt = np.diff(t)
    
    tau = 1
    xi = np.zeros((len(x)))
    dot = np.zeros((len(x)))
    cos = np.zeros((len(x)))
    arcdot = np.zeros(len(x))
    dot = []
    
    for i in range(tau, len(x)-tau):
        
        dd = tau
        a1 = x[i-dd, :]
        a2 = x[i, :]
        a3 = x[i+dd, :]
        d21 = a2 - a1
        d32 = a3 - a2 
        dot.append(np.dot(d21, d32))
        
        arcdot[i] = np.arccos(np.dot(d21, d32) / (np.sqrt(np.sum(d21**2)) * np.sqrt(np.sum(d32**2)))) * 180 / np.pi
        
        cos[i] = np.dot(d21, d32) / (np.sqrt(np.sum(d21**2)) * np.sqrt(np.sum(d32**2)))
        const = 1 - v[i]/vmean
        # psi[i] = const * np.abs(np.dot(d21, d32)) / (t[i+1] - t[i-1])
        
        xi[i] = const * arcdot[i] / (t[i+1] - t[i-1])
        
    angles = np.arccos(cos) * 180 / np.pi    

    return xi[1:-1], angles[1:-1]


# Test
pn = 9
sd = tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]

psi, angles = xi_metric(sd)

# Psi and angle plot
plt.figure()
plt.subplot(2,1,1)
plt.plot(sd['TIME'].values[1:-1],psi)
# plt.xlabel('Time (s)', fontsize=10)
plt.ylabel('$\Xi$ (degrees $s^{-1}$)', fontsize=10)
plt.title('Archea Track', fontsize=10)
plt.grid()

plt.subplot(2,1,2)
plt.plot(sd['TIME'].values[1:-1], angles)
plt.xlabel('Time (s)', fontsize=10)
plt.ylabel('Angles (degrees)', fontsize=10)
# plt.title('Angles', fontsize=10)
plt.grid()


# Track with event detected
fig = pyplot.figure()
ax = Axes3D(fig)

sd =  tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
rec = recs[recs['PARTICLE'] == pn]

ax.plot(sd.X, sd.Y, sd.Z, linewidth=3)
# ax.scatter(rec.X, rec.Y, rec.Z, s=50, marker='o', c='red')


ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (um)', fontsize='18')
ax.set_ylabel('y (um)', fontsize='18')
ax.set_zlabel('z (um)', fontsize='18')
# fig.colorbar(p, ax=ax)
pyplot.show()

#%% Test with LW track
file = '/media/erick/NuevoVol/LINUX_LAP/PhD/Thesis/Results/1/Archea/LW track with reversals/track243_1_0.txt'
sd_lw = np.loadtxt(file)
sd_lw = pd.DataFrame(sd_lw[:,:4], columns=['TIME', 'X' ,'Y', 'Z'])
# sd_lw = pd.DataFrame(sd_lw[:,[0, 4, 5, 6]], columns=['TIME', 'X' ,'Y', 'Z'])

sd_lw_speed = np.array(f.get_speed(sd_lw)).transpose()
sd_lw_speed = pd.DataFrame(sd_lw_speed, columns=['SPEED', 'X', 'Y', 'Z', 'TIME'])


# LW metric
psi, angles = xi_metric(sd)

plt.figure()
plt.subplot(2,1,1)
plt.plot(sd['TIME'].values[1:-1],psi)
# plt.xlabel('Time (s)', fontsize=10)
plt.ylabel('$\Xi$ ($\mu m^{2}s^{-1}$)', fontsize=10)
plt.title('LW Track', fontsize=10)
plt.grid()

plt.subplot(2,1,2)
plt.plot(sd['TIME'].values[1:-1], angles)
plt.xlabel('Time (s)', fontsize=10)
plt.ylabel('Angles (degrees)', fontsize=10)
# plt.title('Angles', fontsize=10)
plt.grid()

#%% RPD
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

sd_lw_speed['PARTICLE'] = np.ones(len(sd_lw_speed))
angles, rec = get_angles_rdp(sd_lw_speed, 10, True)

fig = pyplot.figure()
ax = Axes3D(fig)


ax.plot(sd_lw_speed.X, sd_lw_speed.Y, sd_lw_speed.Z, linewidth=3)
ax.scatter(rec[:, 0], rec[:, 1], rec[:, 2], s=50, marker='o', c='red')


ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (um)', fontsize='18')
ax.set_ylabel('y (um)', fontsize='18')
ax.set_zlabel('z (um)', fontsize='18')
# fig.colorbar(p, ax=ax)
pyplot.show()

#%%
pks, _ = find_peaks(angles, height=1)
plt.plot(t[2:-2], angles[1:-1])
plt.plot(t[1:-1][pks], angles[pks], 'o')
    
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot


fig = pyplot.figure()
ax = Axes3D(fig)

# pn = pnum[1]
# pn = 9

# sd =  tracks_w_speed[tracks_w_speed['PARTICLE'] == pn]
# rec = recs[recs['PARTICLE'] == pn]

p = ax.scatter(sss.X, sss.Y, sss.Z, linewidth=3, c=sss.TIME)
ax.scatter(x[1:-1][pks, 0], x[1:-1][pks, 1], x[1:-1][pks, 2], s=500, marker='o', c='red')


ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (um)', fontsize='18')
ax.set_ylabel('y (um)', fontsize='18')
ax.set_zlabel('z (um)', fontsize='18')
# fig.colorbar(p, ax=ax)
pyplot.show()



#%%
def plotly_scatter(tracks):
    import plotly.express as px
    import plotly
    from plotly.offline import plot, iplot
    
    fig = px.scatter_3d(tracks, x='X', y='Y', z='Z', color='TIME', labels='TIME')
    fig.update_traces(marker=dict(size=1))
    plot(fig)
    # iplot(fig)

plotly_scatter(sd)

