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

from scipy.interpolate import interp1d, interp2d, griddata
from scipy.signal import iirfilter, lfilter, tukey, filtfilt

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

from datetime import datetime, timedelta
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

event_folder = '/kuafu/EventData/Curie'
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

# use the bathymetry to account for the DAS elevation
temp = np.load('/kuafu/EventData/Curie/bathymetry.npz')
bathymetry = temp['bathymetry']
lon = temp['lon']
lat = temp['lat']

# build the interpolation funciton 
lon_grid, lat_grid = np.meshgrid(lon, lat)

f = interp2d(lon, lat, bathymetry, kind='linear')

elevation = [f(das_info.longitude.iloc[i], das_info.latitude.iloc[i]).flatten() for i in range(das_info.shape[0])]
elevation = np.array(elevation).squeeze()
das_info['elevation_m'] = elevation

das_info.to_csv(event_folder + '/das_info.csv', index=False)

# %%
# process raw
data_list = ['/kuafu/DASdata/Curie20060610/LosVilos2',
            '/kuafu/DASdata/Curie20060610/LaLigua',
            '/kuafu/DASdata/Curie20060610/Paihuano',
            '/kuafu/DASdata/Curie20060610/Mendoza',
            '/kuafu/DASdata/Curie20060610/LosAndes',
            '/kuafu/DASdata/Curie20060610/Colina',
            '/kuafu/DASdata/Curie20060610/Valparaiso',
            '/kuafu/DASdata/Curie20060610/LosVilos1']
event_id_list = np.arange(9000, 9008)

# %%
# extract event waveform
data_folder = event_folder + '/data_raw'
mkdir(data_folder)
time_drift = {'9000':12.5, '9001':12.5, '9002':0, '9003':0, '9004':9, '9005':8.5, '9006':9, '9007':9}
i_data = 7
for i_data in [0, 2, 3, 4, 5]:#[1, 6, 7]:#range(len(data_list)):
    print(data_list[i_data])
    event_now = catalog.iloc[i_data, :]
    mg = manager.HDASManager(data_list[i_data] + '/data/')

    # Load traces from bin files
    event_time = datetime.strptime(catalog.iloc[i_data, :].event_time, '%Y-%m-%dT%H:%M:%S.000000Z') - timedelta(seconds=time_drift[str(event_now.event_id)]) 
    start =  event_time - timedelta(seconds=30) 
    end = event_time + timedelta(seconds=90)

    mg.set_time_range(start, end)
    mg.set_data_type('HDAS_2Dmap_Strain')
    data0, time, pos = mg.get_traces()

    dt = mg.param['sample_period']
    das_time0 = np.arange(data0.shape[0])*dt

    # filter above 0.5Hz
    b, a = iirfilter(4, [0.5, 20], btype='bandpass', ftype='butter', fs=1/dt)
    data = filtfilt(b, a, data0, axis=0)

    # Processing traces
    # convert to strain rate
    data = np.diff(data, axis=0)/dt/1e3 # I guess it is 10-3 strain
    das_time0 = das_time0[:-1] + dt/2

    # Downsample from 250 Hz to 100 Hz using interp
    das_dt_ds = 0.01
    das_time_ds = np.arange(das_time0[0], das_time0[-1], das_dt_ds)

    downsample_f = interp1d(das_time0, data, axis=0, bounds_error=False, fill_value=0)
    data_diff_ds = downsample_f(das_time_ds)

    das_time_ds = das_time_ds-30

    # taper the boundary
    taper = tukey(data_diff_ds.shape[0], alpha=0.1)
    taper = taper[:, np.newaxis]
    data_diff_ds = data_diff_ds*taper

    # save the data
    event_data = data_diff_ds
    event_info = {}
    event_info['event_id'] = event_now.event_id
    event_info['event_time'] = event_now.event_time
    event_info['begin_time'] = datetime.strftime(start,'%Y-%m-%dT%H:%M:%S.000000Z')
    event_info['end_time'] = datetime.strftime(end,'%Y-%m-%dT%H:%M:%S.000000Z')
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


#%%
#%% plot event waveform
fig_folder = event_folder + '/event_examples'
mkdir(fig_folder)
tt_output_dir = event_folder + '/theoretical_arrival_time0'

time_drift = {'9000':12.5, '9001':12.5, '9002':0, '9003':0, '9004':9, '9005':8.5, '9006':9, '9007':9}
ymin_list = [0, 0, 0]
for ii in range(len(data_list)):
    event_now = catalog.iloc[ii, :]
    event_id = event_now.event_id
    event_name = event_now.place
    magnitude = event_now.magnitude

    data, info = load_event_data(event_folder + '/data_raw', event_id)
    DAS_channel_num = data.shape[1]
    dt = info['dt_s']
    das_time = np.arange(data.shape[0])*dt-30

    tt_1d = pd.read_csv(tt_output_dir + f'/1D_tt_{event_id}.csv')

    plt.close('all')
    fig, gca = plt.subplots(figsize=(10, 6))
    plot_das_waveforms(data[::10, :], das_time[::10]+time_drift[str(event_id)], gca, pclip=90)
    gca.plot(tt_1d.P_arrival, '--g')
    gca.plot(tt_1d.S_arrival, '-g')
    gca.set_ylim(0, 60)
    gca.set_title(f'M{magnitude}, {event_name} time shift {time_drift[str(event_id)]}s')

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


    plt.savefig(fig_folder + f'/{event_id}_waveforms_raw.png', bbox_inches='tight')

# %%
# compare the waveform filterd to different band
time_drift = {'9000':12.5, '9001':12.5, '9002':0, '9003':0, '9004':9, '9005':8.5, '9006':9, '9007':9}
ymin_list = [0, 0, 0]
ii=1
event_now = catalog.iloc[ii, :]
event_id = event_now.event_id
event_name = event_now.place
magnitude = event_now.magnitude

data1, info1 = load_event_data(event_folder + '/data_raw', event_id)
data2, info2 = load_event_data(event_folder + '/data', event_id)
DAS_channel_num = data1.shape[1]
dt1 = info1['dt_s']
das_time1 = np.arange(data1.shape[0])*dt1-30

dt2 = info2['dt_s']
das_time2 = np.arange(data2.shape[0])*dt2

fig, gca = plt.subplots(figsize=(10, 5))
gca.plot(das_time1, data1[:, 1200], label='[0.5, 20]Hz')
gca.plot(das_time2-8.25, data2[:, 1200], alpha=0.8, label='[1, 10]Hz')
gca.set_xlabel('Time (s)')
gca.set_ylabel('Strain rate ($10^{-6}$)')
gca.set_xlim(-10, 35)
gca.set_title(f'{event_name}, Channel 1200')
gca.legend()
plt.savefig(fig_folder + f'/{event_id}_waveforms_comparison_freq_bands.png', bbox_inches='tight')
# %%
# check the time difference
time_drift = {'9000':12.5, '9001':12.5, '9002':np.nan, '9003':np.nan, '9004':9, '9005':8.5, '9006':9, '9007':9}

event_time_list = event_time = [obspy.UTCDateTime(catalog.iloc[ii, :].event_time) for ii in range(0, 8)]
event_time_diff = [(event_time_list[ii] - event_time_list[-1])/24/3600 for ii in range(0, 8)]
time_drift_list = [time_drift[temp] for temp in time_drift]
plt.plot(event_time_diff, time_drift_list, 'o', markersize=15)
plt.xlabel('Days after the earliest event')
plt.ylabel('Time shift (s)')
plt.savefig(fig_folder + f'/time_shfit_to_match_event_time.png', bbox_inches='tight')

# %%
