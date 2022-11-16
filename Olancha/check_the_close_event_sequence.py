#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser
import obspy
import statsmodels.api as sm

import sys
sys.path.append('../')
from utility.loading import *
from utility.general import mkdir
from utility.processing import remove_outliers, filter_event
from utility.plotting import add_annotate, plot_das_waveforms

import seaborn as sns

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
event_dir  = '/kuafu/EventData/Olancha_old'
olancha_old_results_output_dir = '/kuafu/yinjx/Olancha/peak_ampliutde_scaling_results_strain_rate/Old'
peak_amplitude_df_olancha_old = pd.read_csv(olancha_old_results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')

das_info = pd.read_csv(event_dir + '/das_info.csv')
catalog = pd.read_csv(event_dir + '/catalog.csv')
close_events_pd = pd.read_csv(olancha_old_results_output_dir + '/very_close_events.csv')
catalog_close_events = catalog[catalog.event_id.isin(close_events_pd.event_id)]
# %%
plt.plot(das_info.longitude, das_info.latitude, '-k')
plt.plot(catalog_close_events.longitude, catalog_close_events.latitude, 'r.')
# %%
temp_distance = np.array(peak_amplitude_df_olancha_old[peak_amplitude_df_olancha_old.event_id==38603762].calibrated_distance_in_km)
cb = plt.scatter(das_info.longitude, das_info.latitude,s=1, c= temp_distance, cmap='viridis')
plt.colorbar(cb)
plt.plot(catalog_close_events[catalog_close_events.event_id==38603762].longitude, catalog_close_events[catalog_close_events.event_id==38603762].latitude, 'r.')
# %%
# check waveforms
for ii, event_id in enumerate([38603762]):#enumerate(close_events_pd.event_id):
    data, info = load_event_data(event_dir + '/data', event_id)
    
    das_time = np.arange(0, data.shape[0])*info['dt_s']-30
    picks_P_time, channel_P, picks_S_time, channel_S = load_phasenet_pick(event_dir + '/picks_phasenet_das', event_id, das_time, das_info['index'], time_range=None, include_nan=True)

    i_channel = 3046
    plt.plot(das_time, data[:, i_channel]+ii*30)
    plt.plot(picks_P_time[i_channel] + np.array([0, 2]), ii*30*np.ones(2), '-rx', markersize=10)
    plt.plot(picks_S_time[i_channel] + np.array([0, 2]), ii*30*np.ones(2), '-kx', markersize=20)

    plt.text(0, ii*30, f"M {info['magnitude']:.2f}")
    plt.xlim(0, 10)

#%%
data, info = load_event_data(event_dir + '/data', 39550567) 
fig, gca=plt.subplots(figsize=(10, 8))
plot_das_waveforms(data, das_time, gca, title=None, pclip=99, ymin=0, ymax=15)
# %%
for event_id in catalog_close_events.event_id:
    temp_xx = peak_amplitude_df_olancha_old[peak_amplitude_df_olancha_old.event_id==event_id]
    plt.plot(temp_xx.calibrated_distance_in_km, temp_xx.peak_P, '.', alpha=0.1)
plt.hlines(y=40, xmin=0, xmax=40, color='r')
plt.xlim(10, 20)
plt.yscale('log')
# %%
for event_id in catalog_close_events.event_id:
    temp_xx = peak_amplitude_df_olancha_old[peak_amplitude_df_olancha_old.event_id==event_id]
    plt.plot(temp_xx.calibrated_distance_in_km, temp_xx.peak_S, '.', alpha=0.1)
plt.hlines(y=40, xmin=0, xmax=40, color='r')
plt.xlim(10, 20)
plt.yscale('log')
# %%

#%%
# Check time variation
peak_amplitude_df_temp = peak_amplitude_df_olancha_old.groupby(by=['event_id', 'region'], as_index=False).mean()
event_catalog_info = catalog[catalog.event_id.isin(peak_amplitude_df_olancha_old.event_id.unique())]

temp_df = pd.merge(peak_amplitude_df_temp, event_catalog_info[['event_id', 'event_time']], on=['event_id'])
# %%
time = np.array([obspy.UTCDateTime(event_time) for event_time in np.array(temp_df.event_time)])
time = time - time[0]
time = (time-np.min(time))/3600/24

scaled_amplitude = temp_df.peak_S / 10**(temp_df.magnitude/2) * temp_df.calibrated_distance_in_km**1.4
plt.semilogy(time, scaled_amplitude, '.')
# %%
