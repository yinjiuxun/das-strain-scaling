#%%
import pandas as pd
#from sep_util import read_file
import utm
import numpy as np
import h5py
import time
import tqdm
import obspy
import datetime
import os
import random

import sys
sys.path.append('../')
from utility.general import *
from utility.loading import load_phasenet_pick, load_event_data

from eew_utility import get_DAS_response_time, get_SEIS_response_time

from obspy.taup import TauPyModel
from scipy.interpolate import interp2d, griddata

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %matplotlib inline
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 18, # fontsize for x and y labels (was 10)
    'axes.titlesize': 18,
    'font.size': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex':False,
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
}
matplotlib.rcParams.update(params)

#%%
# Define the path to store all the output results
output_dir = '/kuafu/yinjx/EEW_general/ideal_response_time'
mkdir(output_dir)

fig_output_dir = '/kuafu/yinjx/EEW_general/ideal_response_time/figures'
mkdir(fig_output_dir)
#%%|
# make the evaluation domain (earthquake can be at every grid point)
eq_depth = 20
nx, ny = 11, 41
eq_domain_x = np.linspace(-110, 0, nx, endpoint=False)
eq_domain_y = np.linspace(-100, 100, ny, endpoint=True)

# %%
# make the synthetic das array
das_label = 'Cshape' #['horizontal', 'vertical', 'cross', 'Cshape', 'multiple_arrays']

if das_label == 'horizontal':
    n_channel = 5000
    dx_das = 20*1e-3
    das_x = -np.arange(0, n_channel)*dx_das
    das_y = np.zeros(das_x.shape)
    das_wave_type = 'S'

elif das_label == 'vertical':
    n_channel = 5000
    dx_das = 20*1e-3
    das_y = np.arange(-n_channel/2, n_channel/2)*dx_das
    das_x = np.zeros(das_y.shape)-50
    das_wave_type = 'S'

elif das_label == 'cross':
    n_channel = 5000
    dx_das = 20*1e-3
    das_x1 = -np.arange(0, n_channel)*dx_das
    das_y1 = np.zeros(das_x1.shape)
    das_y2 = np.arange(-n_channel/2, n_channel/2)*dx_das
    das_x2 = np.zeros(das_y2.shape)-50
    das_x = np.concatenate([das_x1, das_x2], axis=0)
    das_y = np.concatenate([das_y1, das_y2], axis=0)
    das_wave_type = 'P'


elif das_label == 'Cshape':
    n_channel = 5000
    dx_das = 20*1e-3
    das_y1 = np.arange(-n_channel/2, n_channel/2)*dx_das
    das_x1 = np.zeros(das_y1.shape)-50
    das_x2 = np.array([-np.arange(0, n_channel/2)*dx_das, -np.arange(0, n_channel/2)*dx_das]).flatten()
    das_y2 = np.array([np.zeros(int(n_channel/2))+50, np.zeros(int(n_channel/2))-50]).flatten()

    das_x = np.concatenate([das_x1, das_x2], axis=0)
    das_y = np.concatenate([das_y1, das_y2], axis=0)
    das_wave_type = 'P'

elif das_label == 'multiple_arrays':
    n_channel = 5000
    dx_das = 20*1e-3
    n_das = 5
    das_x0 = np.repeat([-np.arange(0, n_channel)*dx_das], n_das, axis=0)
    das_y0 = np.repeat([np.ones(das_x0.shape[1])], n_das, axis=0) * np.linspace(-100, 100, n_das)[:, np.newaxis]
    das_x = das_x0.flatten()
    das_y = das_y0.flatten()
    das_wave_type = 'P'

else:
    raise NameError('das_label unknown or undefined')

#%%
# make the station distribution 
dx_st, dy_st = 10, 10
st_domain_x = np.arange(0, 50+dx_st, dx_st)
st_domain_y = np.arange(-100, 100+dy_st, dy_st)
#%%
# generate the source grid
eq_x_grid, eq_y_grid = np.meshgrid(eq_domain_x, eq_domain_y)

# generate the station grid
st_x_grid, st_y_grid = np.meshgrid(st_domain_x, st_domain_y)

#%%
# map of source, station, and DAS
plt.plot(eq_x_grid, eq_y_grid, 'k+')
plt.plot(das_x, das_y, 'r.')
plt.plot(st_x_grid, st_y_grid, 'kv')

#%%
# First look for the precalculated TTT, if not exists, get one from interpolating TauP 
travel_time_table_file = output_dir + '/travel_time_table.npz'
if not os.path.exists(travel_time_table_file):
    model = TauPyModel(model='iasp91')

    # distance list
    distance_fit = np.linspace(0, 5, 200)
    # depth list
    depth_fit = np.arange(10, 60, 5)

    distance_grid, depth_grid = np.meshgrid(distance_fit, depth_fit)


    tavel_time_P_grid = np.zeros(distance_grid.shape)
    tavel_time_S_grid = np.zeros(distance_grid.shape)

    #for i_eq in tqdm.tqdm(range(10), desc="Calculating arrival time..."):
    for i_depth in tqdm(range(depth_grid.shape[0]), desc="Calculating arrival time..."):   

        for i_distance in range(distance_grid.shape[1]):
            try:
                arrivals = model.get_ray_paths(depth_fit[i_depth], distance_fit[i_distance], phase_list=['p', 's'])
                tavel_time_P_grid[i_depth, i_distance] = arrivals[0].time
                tavel_time_S_grid[i_depth, i_distance] = arrivals[1].time 
            except:
                tavel_time_P_grid[i_depth, i_distance] = np.nan
                tavel_time_S_grid[i_depth, i_distance] = np.nan

    # save the calculated Travel time table
    np.savez(travel_time_table_file, distance_fit=distance_fit, depth_fit=depth_fit, 
             tavel_time_P_grid=tavel_time_P_grid, tavel_time_S_grid=tavel_time_S_grid)

    print('Travel time table calculated!')
    
#%%    
# The TTT calculated or already exists, directly load it.
temp = np.load(travel_time_table_file)
distance_fit = temp['distance_fit']
depth_fit = temp['depth_fit']
tavel_time_P_grid = temp['tavel_time_P_grid']
tavel_time_S_grid = temp['tavel_time_S_grid']
distance_grid, depth_grid = np.meshgrid(distance_fit, depth_fit)

#%%
# now begin to calculate time
#%%
# now begin to calculate time
# for eq_depth in [20]:#[10, 20, 30, 40, 50]:
eq_depth = 20
if das_wave_type == 'P':
    DAS_response_time = get_DAS_response_time(eq_depth, das_x, das_y, eq_x_grid, eq_y_grid, distance_grid, depth_grid, tavel_time_P_grid)
elif das_wave_type == 'S':
    DAS_response_time = get_DAS_response_time(eq_depth, das_x, das_y, eq_x_grid, eq_y_grid, distance_grid, depth_grid, tavel_time_S_grid)

STA_response_time = get_SEIS_response_time(eq_depth, eq_x_grid, eq_y_grid, st_x_grid, st_y_grid, distance_grid, depth_grid, tavel_time_P_grid)


DAS_response_time = np.reshape(DAS_response_time, eq_x_grid.shape)
STA_response_time = np.reshape(STA_response_time, eq_x_grid.shape)
diff_response_time = DAS_response_time-STA_response_time

#%%
# plotting 
cmap = plt.cm.get_cmap('BrBG_r', 16)
fig, ax = plt.subplots(figsize=(8, 10))
clb=ax.imshow(diff_response_time, 
extent=[eq_domain_x.min(), eq_domain_x.max(), eq_domain_y.min(), eq_domain_y.max()],
cmap=cmap, vmin=-10, vmax=10)
ax.contour(eq_x_grid, eq_y_grid, diff_response_time, levels=[0])

ax.plot(das_x, das_y, 'r.')
ax.plot(st_x_grid, st_y_grid, 'kv')
ax.vlines(x=st_domain_x.min()-2, ymax=eq_domain_y.max(), ymin=eq_domain_y.min(), color='gray', linestyle='--')
ax.vlines(x=st_domain_x.min()-100, ymax=eq_domain_y.max(), ymin=eq_domain_y.min(), color='gold', linestyle='-.', linewidth=3)


ax.text(eq_domain_x.mean()-50, eq_domain_y.max()+10, 'Offshore region')
ax.text(st_domain_x.min()+20, eq_domain_y.max()+10, 'Land region')
ax.text(das_x.mean()-10, das_y.mean()+10, 'DAS', fontdict={'color':'red'})
ax.text(eq_domain_x.min()+15, eq_domain_y.max()-20, f'source depth:\n{eq_depth}km', bbox=dict(boxstyle="round", fc="white"))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

ax.set_xlabel('Distance along trench (km)')
ax.set_ylabel('Distance along coastline (km)')

plt.colorbar(clb, cax=cax, label='response time difference')
plt.savefig(fig_output_dir + f'/{das_label}_DAS_depth_{eq_depth}km_das_{das_wave_type}.png', bbox_inches='tight')
# %%
