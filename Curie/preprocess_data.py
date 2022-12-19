#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser
import obspy
import statsmodels.api as sm
import sys 
import h5py
import tqdm

from scipy.interpolate import interp1d

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.loading import save_rawevent_h5, load_event_data
from utility.processing import remove_outliers, filter_event
from utility.plotting import plot_das_waveforms

import seaborn as sns

# tool specifically for Curie data
from hdas import manager
import preproc


# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %matplotlib inline
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 100,  # to adjust notebook inline plot size
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


event_folder = '/kuafu/EventData/Curie'  #'/kuafu/EventData/AlumRock5.1/MammothNorth'#'/kuafu/EventData/Ridgecrest' 
tt_output_dir = event_folder + '/theoretical_arrival_time0'
# %%
# process the catalog 
catalog0 = pd.read_csv(event_folder + '/event_info.csv')
catalog0['event_id'] = catalog0.index + 9000
catalog0.to_csv(event_folder + '/event_info.csv', index=False)

catalog = pd.DataFrame()
catalog['event_id'] = catalog0['event_id']
catalog['event_time'] = [str(obspy.UTCDateTime(temp)) for temp in catalog0['Local Time']]
catalog['latitude'] = catalog0['Lat']
catalog['longitude'] = catalog0['Lon']
catalog['depth_km'] = catalog0['Depth']
catalog['magnitude'] = catalog0['Magnitude']
catalog['magnitude_type'] = catalog0['Magnitude_type'].str.lower()
catalog['source'] = 'chile'
catalog['place'] = catalog0['Place']
catalog.to_csv(event_folder + '/catalog.csv', index=False)

# %%
# processing the channel info
curie_coordinate = pd.read_csv(event_folder + '/curie_coordinates.csv')
das_info = pd.DataFrame()
das_info['index'] = curie_coordinate.index
das_info['latitude'] = curie_coordinate['lat']
das_info['longitude'] = curie_coordinate['lon']
das_info['elevation_m'] = 0
das_info.to_csv(event_folder + '/das_info.csv', index=False)

# %%
# calculate travel time
event_list = [9001, 9006, 9007]

# Work out a handy travel time table to do interpolation
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

event_folder = event_folder
# load the DAS array information
DAS_info = pd.read_csv(event_folder + '/das_info.csv')
catalog = pd.read_csv(event_folder + '/catalog.csv')
eq_num = len(catalog)
eq_lat = catalog.latitude # lat
eq_lon = catalog.longitude # lon
eq_mag = catalog.magnitude # catalog magnitude
eq_time = catalog.event_time # eq time


DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info.index
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

# calculate tt with taup
mkdir(tt_output_dir)
travel_time_table_file = event_folder + '/travel_time_table.npz'

# from one event to all channels
event_arrival_P = np.zeros((DAS_channel_num, eq_num)) 
event_arrival_S = np.zeros((DAS_channel_num, eq_num)) 

# First look for the precalculated TTT, if not exists, get one from interpolating TauP 
if not os.path.exists(travel_time_table_file):
    model = TauPyModel(model='iasp91')

    # distance list
    distance_fit = np.linspace(0, 3, 100)
    # depth list
    depth_fit = np.arange(0, 200, 2)

    distance_grid, depth_grid = np.meshgrid(distance_fit, depth_fit)


    tavel_time_P_grid = np.zeros(distance_grid.shape)
    tavel_time_S_grid = np.zeros(distance_grid.shape)

    #for i_eq in tqdm.tqdm(range(10), desc="Calculating arrival time..."):
    for i_depth in tqdm.tqdm(range(depth_grid.shape[0]), desc="Calculating arrival time..."):   

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

#%%
# build the interpolation function
from scipy.interpolate import interp2d, griddata
#grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

ii = ~np.isnan(tavel_time_p_grid) # ignore the nan
# interp_f_p = interp2d(distance_grid[ii], depth_grid[ii], tavel_time_p_grid[ii], kind='linear')
# interp_f_s = interp2d(distance_grid[ii], depth_grid[ii], tavel_time_s_grid[ii], kind='linear')

for i_eq in tqdm.tqdm(range(eq_num), desc="Calculating arrival time..."):   
    # estimate the arrival time of each earthquake to all channels
    P_arrival = np.zeros(DAS_channel_num)
    S_arrival = np.zeros(DAS_channel_num)
    distance_to_source = locations2degrees(DAS_lat, DAS_lon, catalog.iloc[i_eq, :].latitude, catalog.iloc[i_eq, :].longitude)

    P_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_p_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*catalog.iloc[i_eq, :].depth_km))
    S_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_s_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*catalog.iloc[i_eq, :].depth_km))

    DAS_info['P_arrival'] = P_arrival
    DAS_info['S_arrival'] = S_arrival

    DAS_info.to_csv(tt_output_dir + f'/1D_tt_{catalog.iloc[i_eq, :].event_id}.csv', index=False)
# %%
# extract event waveform
data_folder = event_folder + '/data'
mkdir(data_folder)

# load the data and look at the waveform
fs = 250
event_id_list = [9001, 9006, 9007]

for ii, i_event in enumerate([1, 6, 7]): #
    event_now = catalog.iloc[i_event, :]
    # event_path = event_folder + f'/h5_data/{event_now.event_id}.h5'
    # with h5py.File(event_path, 'r') as fid:
    #     data = fid['data'][:].T

    event_path = event_folder + f'/npz/{event_now.event_id}.npz'
    temp = np.load(event_path)
    data = temp['data']

    dt = 1/fs
    das_time0 = np.arange(data.shape[0])*dt

    # convert to strain rate
    data = np.diff(data, axis=0)/dt/1e3 # I guess it is 10-3 strain
    das_time0 = das_time0[:-1] + dt/2

    # Downsample from 250 Hz to 100 Hz using interp
    das_dt_ds = 0.01
    das_time_ds = np.arange(das_time0[0], das_time0[-1], das_dt_ds)

    downsample_f = interp1d(das_time0, data, axis=0, bounds_error=False, fill_value=0)
    data_diff_ds = downsample_f(das_time_ds)

    event_data = data_diff_ds
    event_info = {}
    event_info['event_id'] = event_now.event_id
    event_info['event_time'] = event_now.event_time
    event_info['begin_time'] = event_now.event_time
    event_info['end_time'] = event_now.event_time
    event_info['latitude'] = event_now.latitude
    event_info['longitude'] = event_now.longitude
    event_info['depth_km'] = event_now.depth_km
    event_info['magnitude'] = event_now.magnitude
    event_info['magnitude_type'] = event_now.magnitude_type
    event_info['source'] = event_now.source
    event_info['dt_s'] = das_dt_ds
    event_info['dx_m'] = 5
    event_info['das_array'] = 'Curie'

    save_rawevent_h5(os.path.join(data_folder, str(event_now.event_id)+'.h5'), event_data, event_info)

#%% plot event waveform
fig_folder = event_folder + '/event_examples'
mkdir(fig_folder)
tt_output_dir = event_folder + '/theoretical_arrival_time_calibrated'

# # correct 
# tt_shift_list = [-4.5, 25.5, -10.5]
# tt_1d.P_arrival = tt_1d.P_arrival+tt_shift_list[ii]
# tt_1d.S_arrival = tt_1d.S_arrival+tt_shift_list[ii]
# tt_1d.to_csv(event_folder + '/theoretical_arrival_time_calibrated' + f'/1D_tt_{event_id}.csv')


ymin_list = [0, 0, 0]
for ii, i_event in enumerate([1, 6, 7]):
    event_now = catalog.iloc[i_event, :]
    event_id = event_now.event_id
    event_name = event_now.place
    magnitude = event_now.magnitude

    data, info = load_event_data(event_folder + '/data', event_id)
    DAS_channel_num = data.shape[1]
    dt = info['dt_s']
    das_time = np.arange(data.shape[0])*dt

    tt_1d = pd.read_csv(tt_output_dir + f'/1D_tt_{event_id}.csv')

    fig, gca = plt.subplots(figsize=(10, 6))
    plot_das_waveforms(data[::10, :], das_time[::10], gca, pclip=95)
    gca.plot(tt_1d.P_arrival, '--g')
    gca.plot(tt_1d.S_arrival, '-g')
    gca.set_ylim(ymin_list[ii], ymin_list[ii]+60)
    gca.set_title(f'M{magnitude}, {event_name}')

    # try:
    #     ML_picking_dir = event_folder + '/picks_phasenet_das'
    #     tt_tp = np.zeros(shape=DAS_channel_num)*np.nan
    #     tt_ts = tt_tp.copy()

    #     ML_picking = pd.read_csv(ML_picking_dir + f'/{event_id}.csv')
    #     ML_picking = ML_picking[ML_picking.channel_index < DAS_channel_num]

    #     ML_picking_P = ML_picking[ML_picking.phase_type == 'P']
    #     ML_picking_S = ML_picking[ML_picking.phase_type == 'S']
    #     # ML_picking_P = remove_ml_tt_outliers(ML_picking_P, das_dt, tdiff=25, given_range=given_range_P)
    #     # ML_picking_S = remove_ml_tt_outliers(ML_picking_S, das_dt, tdiff=25, given_range=given_range_S)

    #     tt_tp[ML_picking_P.channel_index] = das_time[ML_picking_P.phase_index]
    #     tt_ts[ML_picking_S.channel_index] = das_time[ML_picking_S.phase_index]
    #     gca.plot(tt_tp, '--k', linewidth=2, label='ML-picked P')
    #     gca.plot(tt_ts, '-k', linewidth=2, label='ML-picked S')
    # except:
    #     print('Cannot find the ML travel time, skip...')


    plt.savefig(fig_folder + f'/{event_id}_waveforms.png', bbox_inches='tight')


# %%
