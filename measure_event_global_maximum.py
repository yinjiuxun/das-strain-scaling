#%%
# Import modules
import pandas as pd
#from sep_util import read_file
import utm
import numpy as np
import h5py
import dateutil
import time
import tqdm
import obspy
import datetime
import os
import glob
import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

import warnings

from utility_functions import *
from obspy.geodetics import locations2degrees

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

# %%
# Setup the paths
event_folder_list = ['/kuafu/EventData/Ridgecrest', '/kuafu/EventData/Mammoth_north', '/kuafu/EventData/Mammoth_south']
peak_amplitude_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events']

for ii_region in [2]:# [0, 1, 2]:
    event_folder, peak_amplitude_dir = event_folder_list[ii_region], peak_amplitude_dir_list[ii_region]

    print('='*10 + peak_amplitude_dir + '='*10)

    # Event waveform directory, phase picking directory
    data_path = event_folder + '/data'
    pick_path = event_folder + '/picks_phasenet_das' # directory of ML phase picker

    # Load the catalog, filter events, load waveforms
    catalog = pd.read_csv(event_folder + '/catalog.csv')
    n_eq = catalog.shape[0]
    das_info = pd.read_csv(event_folder + '/das_info.csv')
    n_channel = das_info.shape[0]

    array_maximum = np.zeros((n_eq, n_channel))

    for i_eq in tqdm(range(n_eq)):
        try:
            event_now = catalog.iloc[i_eq, :]
            event_data, event_info = load_event_data(data_path, event_now.event_id)

            if event_data.shape[1] > das_info.shape[0]:
                event_data = event_data[:, das_info.index]

            array_maximum[i_eq, :] = np.nanmax(event_data, axis=0)
        except:
            continue
    np.savez(peak_amplitude_dir + '/global_maximum.npz', array_maximum=array_maximum)

# %%
# Setup the paths
event_folder_list = ['/kuafu/EventData/Ridgecrest', '/kuafu/EventData/Mammoth_north', '/kuafu/EventData/Mammoth_south']
peak_amplitude_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events']

region_label = ['Ridgecrest', 'Mammoth-N', 'Mammoth-S']
fig, ax = plt.subplots(2, 1, figsize=(10, 12))
for ii_region in [0, 1, 2]:

    event_folder, peak_amplitude_dir = event_folder_list[ii_region], peak_amplitude_dir_list[ii_region]

    catalog = pd.read_csv(event_folder + '/catalog.csv')
    ii_M2 = catalog.magnitude >=2

    temp = np.load(peak_amplitude_dir + '/global_maximum.npz')
    temp = temp['array_maximum']
    temp = temp[ii_M2, :]
    channel_max = np.nanmax(temp, axis=0)
    global_max = temp[temp>0].flatten()

    gca = ax[0]
    gca.hist(np.log10(global_max), range=(-4,8), bins=100, label=region_label[ii_region], alpha=0.5)
    gca.set_yscale('log')
    gca.set_ylim(ymin=1)

    gca = ax[1]
    gca.semilogy(np.linspace(0, 1000, len(channel_max)), channel_max, label=region_label[ii_region])

ax[0].legend()
ax[0].set_xlabel('log10(E) (micro strain/s)')
ax[0].set_ylabel('Counts')

ax[1].legend()

plt.savefig('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/maximum_amplitude_distribution.png', bbox_inches='tight')
# %%
