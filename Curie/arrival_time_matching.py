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

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
output_dir = '/kuafu/yinjx/Curie/arrival_time_matching'
mkdir(output_dir)

#%%
# load the DAS channel location
event_folder = '/kuafu/EventData/Curie'
das_info = pd.read_csv(event_folder + '/das_info.csv')
catalog = pd.read_csv(event_folder + '/catalog.csv')

DAS_channel_num = das_info.shape[0]
DAS_index = das_info['index']
DAS_lon = das_info['longitude']
DAS_lat = das_info['latitude']

center_lon = np.mean(DAS_lon)
center_lat = np.mean(DAS_lat)

# %%
# prepare the "grid" point around the das array
n_xgrid, n_ygrid = 75, 75
num_points = n_xgrid * n_ygrid
x_min, x_max = center_lon-1.5, center_lon+1.5
y_min, y_max = center_lat-1.5, center_lat+1.5
xgrid_list = np.linspace(x_min, x_max, n_xgrid, endpoint=True)
ygrid_list = np.linspace(y_min, y_max, n_ygrid, endpoint=True)

lon_grid, lat_grid = np.meshgrid(xgrid_list, ygrid_list)

# %%
fig, ax = plt.subplots(figsize=(7, 6))
cmp = ax.scatter(DAS_lon, DAS_lat, s=10, c=DAS_index, cmap='jet')
ax.plot(center_lon, center_lat, 'r*')
ax.plot(lon_grid.flatten(), lat_grid.flatten(), 'k+')
ax.set_ylim(center_lat-1.1, center_lat+1.1)
ax.set_xlim(center_lon-1.1, center_lon+1.1)
fig.colorbar(cmp)

# %%
# Work out a handy travel time table to do interpolation
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

travel_time_table_file = output_dir + '/travel_time_table.npz'

# from one event to all channels
event_arrival_P = np.zeros((DAS_channel_num, num_points)) 
event_arrival_S = np.zeros((DAS_channel_num, num_points)) 
S_P_diff = np.zeros((DAS_channel_num, num_points)) 
distance_to_source_all = np.zeros((DAS_channel_num, num_points)) 

# First look for the precalculated TTT, if not exists, get one from interpolating TauP 
if not os.path.exists(travel_time_table_file):
    model = TauPyModel(model='iasp91')

    # distance list
    distance_fit = np.linspace(0, 2, 100)
    # depth list
    depth_fit = np.arange(0, 100, 1)

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
    np.savez(travel_time_table_file, distance_grid=distance_grid, depth_grid=depth_grid, 
             tavel_time_p_grid=tavel_time_P_grid, tavel_time_s_grid=tavel_time_S_grid)

    print('Travel time table calculated!')
    
    
# The TTT calculated or already exists, directly load it.
temp = np.load(travel_time_table_file)
distance_grid = temp['distance_grid']
depth_grid = temp['depth_grid']
tavel_time_p_grid = temp['tavel_time_p_grid']
tavel_time_s_grid = temp['tavel_time_s_grid']


# save the calculated Travel time curves as templates
arrival_time_curve_template_file = output_dir + '/arrival_time_template.npz'
assumed_depth = 20
if not os.path.exists(arrival_time_curve_template_file):
    # build the interpolation function
    from scipy.interpolate import interp2d, griddata
    #grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

    ii = ~np.isnan(tavel_time_p_grid) # ignore the nan
    # interp_f_p = interp2d(distance_grid[ii], depth_grid[ii], tavel_time_p_grid[ii], kind='linear')
    # interp_f_s = interp2d(distance_grid[ii], depth_grid[ii], tavel_time_s_grid[ii], kind='linear')


    for i_eq in tqdm(range(num_points), desc="Calculating arrival time..."): 

            # estimate the arrival time of each earthquake to all channels
            P_arrival = np.zeros(DAS_channel_num)
            S_arrival = np.zeros(DAS_channel_num)
            distance_to_source = locations2degrees(DAS_lat, DAS_lon, lat_grid.flatten()[i_eq], lon_grid.flatten()[i_eq])

            P_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_p_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*assumed_depth))
            S_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_s_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*assumed_depth))

            distance_to_source_all[:, i_eq] = distance_to_source
            event_arrival_P[:, i_eq] = P_arrival
            event_arrival_S[:, i_eq] = S_arrival
            S_P_diff[:, i_eq] = S_arrival - P_arrival

    # save the calculated Travel time curves as templates
    np.savez(arrival_time_curve_template_file, lat_grid=lat_grid, lon_grid=lon_grid, distance_grid = distance_to_source_all,
                event_arrival_P=event_arrival_P, event_arrival_S=event_arrival_S, S_P_diff=S_P_diff)

#%%
# Load the calculated Travel time curves as templates
arrival_time_curve_template_file = output_dir + '/arrival_time_template.npz'
temp = np.load(arrival_time_curve_template_file)
distance_to_source_all = temp['distance_grid']
event_arrival_P_template = temp['event_arrival_P']
event_arrival_P_template_diff = np.diff(event_arrival_P_template, axis=0)
event_arrival_S_template = temp['event_arrival_S']
event_arrival_S_template_diff = np.diff(event_arrival_S_template, axis=0)
S_P_diff = event_arrival_S_template - event_arrival_P_template
# %%
# match the arrival time
from numpy.linalg import norm

def match_arrival(event_arrival_observed, event_arrival_template, misfit_type='l1', demean=True):
    ii_nan = np.isnan(event_arrival_observed)
    event_arrival = event_arrival_observed[~ii_nan, np.newaxis]
    template = event_arrival_template[~ii_nan, :]

    if demean:
        # remove mean
        event_arrival = event_arrival - np.mean(event_arrival, axis=0, keepdims=True)
        template = template - np.mean(template, axis=0, keepdims=True)

    if misfit_type == 'l1':
        norm_diff = np.nanmean(abs(event_arrival - template), axis=0) # L1 norm
    elif misfit_type == 'l2':
        norm_diff = np.sqrt(np.nanmean((event_arrival - template)**2, axis=0)) # L2 norm
        
    ii_min = np.nanargmin(norm_diff)

    return ii_min, norm_diff

def misfit_to_probability(misfit):
    # sigma_misfit = np.nanstd(misfit)
    # probability = np.exp(-misfit**2/2/sigma_misfit**2)/sigma_misfit/np.sqrt(2*np.pi)
    # probability = probability/np.nansum(probability)

    probability = 1/misfit/np.nansum(1/misfit)

    return probability

#%%
# test on an event
use_ML_picking = True
if use_ML_picking:
    figure_label = 'ML'
else:
    figure_label = 'VM'

event_id = 9006
event_info = catalog[catalog.event_id == event_id]

tt_output_dir = event_folder + '/theoretical_arrival_time0'
event_arrival_pd = pd.read_csv(tt_output_dir + f'/1D_tt_{event_id}.csv')
event_arrival_P_obs = np.array(event_arrival_pd.P_arrival)
event_arrival_S_obs = np.array(event_arrival_pd.S_arrival)
event_S_P_diff_obs = event_arrival_S_obs - event_arrival_P_obs

if use_ML_picking:
    tt_output_dir = event_folder + '/picks_phasenet_das'
    # Load the DAS data
    strain_rate, info = load_event_data(event_folder+'/data_raw', event_id)
    das_dt = info['dt_s']
    nt = strain_rate.shape[0]
    das_time = np.arange(nt) * das_dt-30
    time_range = (event_arrival_P_obs.min()+25, event_arrival_P_obs.max()+35, event_arrival_S_obs.min()+25, event_arrival_S_obs.max()+35)
    pick_P, channel_P, pick_S, channel_S = load_phasenet_pick(tt_output_dir, event_id, das_time, das_info['index'], time_range=time_range, include_nan=True)
    event_arrival_P_obs = pick_P
    event_arrival_S_obs = pick_S
    event_S_P_diff_obs = pick_S - pick_P

# try to add noise
#noise = 0.1*np.random.normal(size=event_arrival_P_obs.shape)
noise = 0.*np.random.rand(event_arrival_P_obs.shape[0])-0.
event_arrival_P_obs = event_arrival_P_obs + noise
# random discard some points
n_discard = 0
ii_discard = random.sample(range(0, len(event_arrival_P_obs)),n_discard)
event_arrival_P_obs[ii_discard] = np.nan

# random discard some points
n_outlier = 0
ii_outlier = random.sample(range(0, len(event_arrival_P_obs)),n_outlier)
event_arrival_P_obs[ii_outlier] = event_arrival_P_obs[ii_outlier] + 1*np.random.normal(size=n_outlier)

# only use the first 6000 channels
ii_channel = range(0, 7000)
# ii_channel = range(0, DAS_channel_num)

# Fitting
_, norm_diff1 = match_arrival(event_arrival_P_obs[ii_channel], event_arrival_P_template[ii_channel, :], misfit_type='l1')
_, norm_diff2 = match_arrival(event_arrival_S_obs[ii_channel], event_arrival_S_template[ii_channel, :], misfit_type='l1')
_, norm_diff_SP = match_arrival(event_S_P_diff_obs[ii_channel], S_P_diff[ii_channel, :], misfit_type='l1', demean=False)

probability1 = misfit_to_probability(norm_diff1)
probability2 = misfit_to_probability(norm_diff2)
probability_SP = misfit_to_probability(norm_diff_SP)

#%%
# interference
probability = probability1*probability_SP
phase_label = 'P+S-P'
# probability = probability_SP
# phase_label = 'S-P'
# probability = probability1
# phase_label = 'P'
# probability = probability2
# phase_label = 'S'

ii_min = np.nanargmax(probability)


fig, ax = plt.subplots(3, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
ax[0].plot(DAS_lon, DAS_lat, '-b', label='DAS array', zorder=10)
#ax[0].plot(center_lon, center_lat, 'b^')
#ax[0].plot(lon_grid.flatten(), lat_grid.flatten(), 'k+')
ax[0].plot(lon_grid.flatten()[ii_min], lat_grid.flatten()[ii_min], 'o', color='gold', label='optimal location', markersize=15, markeredgewidth=2, markerfacecolor='None')
ax[0].plot(event_info.longitude, event_info.latitude, 'r*', markersize=20, label='catalog location')
ax[0].set_ylim(center_lat-1.5, center_lat+1.5)
ax[0].set_xlim(center_lon-1.5, center_lon+1.5)
# ax[0].set_ylim(-32.7, -32.2)
# ax[0].set_xlim(-72.1, -71.8)

# cbar = ax[0].scatter(lon_grid, lat_grid, s=30, c= np.reshape(probability, lon_grid.shape), cmap='gray', marker='s')
cbar = ax[0].imshow(np.reshape(probability, (75,75)), aspect='auto', extent=[lon_grid.min(), lon_grid.max(), lat_grid.max(), lat_grid.min()])
# ax[0].invert_yaxis()
# ax[0].plot(lon_grid.flatten()[ii_min], lat_grid.flatten()[ii_min], 'yx', markersize=20, markeredgewidth=5)
ax[0].set_title('Earthquake location')
ax[0].legend(fontsize=10, loc=2)
fig.colorbar(cbar, ax=ax[0], label='pdf')


ax[1].plot(event_arrival_P_obs[ii_channel] - np.nanmean(event_arrival_P_obs[ii_channel]), '.r', label='observed P', markersize=3)
ax[1].plot(event_arrival_P_template[ii_channel, ii_min] - np.nanmean(event_arrival_P_template[ii_channel, ii_min]), '-r', label='Matched P')
ax[1].plot(event_arrival_S_obs[ii_channel] - np.nanmean(event_arrival_S_obs[ii_channel]), '.b', label='observed S', markersize=3)
ax[1].plot(event_arrival_S_template[ii_channel, ii_min] - np.nanmean(event_arrival_S_template[ii_channel, ii_min]), '-b', label='Matched S')
# ax[1].plot(event_S_P_diff_obs, '.k', label='observed tS - tP', markersize=3)
# ax[1].plot(S_P_diff[:, ii_min], '-k', label='Matched tS - tP')
ax[1].set_xlabel('Channels')
ax[1].set_ylabel('Demeaned \narrival time (s)')
# ax[1].set_ylim(-2.5, 3)
ax[1].legend(fontsize=10)
origin_time_diff = np.nanmean(event_arrival_P_obs[ii_channel] - event_arrival_P_template[ii_channel, ii_min])
if phase_label == 'P+S-P':
    ax[1].set_title(f'Origin time error: {origin_time_diff:.2f} s')

ax[2].plot(event_S_P_diff_obs, '.k', label='observed tS - tP', markersize=3)
ax[2].plot(S_P_diff[:, ii_min], '-k', label='Matched tS - tP')
ax[2].set_xlabel('Channels')
ax[2].set_ylabel('Arrival time \ndifference (s)')
# ax[1].set_ylim(-2.5, 3)
ax[2].legend(fontsize=10)

plt.savefig(output_dir + f'/event_{event_id}_arrival_matching_{figure_label}_{phase_label}.png', bbox_inches='tight')

# %%
