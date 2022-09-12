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
import dateutil
import pandas as pd

from scipy.interpolate import interp1d
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

def filter_by_channel_number(peak_amplitude_df, min_channel):
    """To remove the measurements from few channels (< min_channel)"""
    event_channel_count = peak_amplitude_df.groupby(['event_id'])['event_id'].count()
    channel_count = event_channel_count.values
    event_id = event_channel_count.index
    event_id = event_id[channel_count >= min_channel]

    return peak_amplitude_df[peak_amplitude_df['event_id'].isin(event_id)]
# %%
min_channel = 100
nearby_channel_number = 10

results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
print(peak_amplitude_df.shape)
peak_amplitude_df = filter_by_channel_number(peak_amplitude_df, min_channel)
print(peak_amplitude_df.shape)
peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrP >=10]
print(peak_amplitude_df.shape)

# %% 
# LA Google data
min_channel = 100
nearby_channel_number = 10

results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
peak_amplitude_df_LA = pd.read_csv(results_output_dir + f'/peak_amplitude_events/calibrated_peak_amplitude.csv')
peak_amplitude_df_LA = filter_by_channel_number(peak_amplitude_df_LA, min_channel)
peak_amplitude_df_LA = peak_amplitude_df_LA[peak_amplitude_df_LA.snrS >=10]
print(peak_amplitude_df_LA.shape)

# %%
cmap = plt.cm.get_cmap('Spectral_r', 20)
fig, ax = plt.subplots(2,2, figsize=(10, 10), sharex=True, sharey=True)
ax = ax.flatten()

region_labels = ['ridgecrest', 'mammothN', 'mammothS', 'LAX']

for region_i in range(4):
    if region_i < 3:
        region_label = region_labels[region_i]
        peak_amplitude_df_now = peak_amplitude_df[peak_amplitude_df.region.str.contains(region_label)]
    else:
        region_label = region_labels[region_i]
        peak_amplitude_df_now = peak_amplitude_df_LA.copy()
        peak_amplitude_df_now.peak_S = peak_amplitude_df_now.peak_S

    # max_channel_id = peak_amplitude_df_now.channel_id.max()
    # channel_downsample = np.arange(0, max_channel_id, max_channel_id//10)
    # peak_amplitude_df_now = peak_amplitude_df_now[peak_amplitude_df_now.channel_id.isin(channel_downsample)]

    peak_amplitude_df_now = peak_amplitude_df_now.sort_values(by=['magnitude'])

    
    gca=ax[region_i]
    clb = gca.scatter(np.log10(peak_amplitude_df_now.distance_in_km), 
                    np.log10(peak_amplitude_df_now.peak_S), 
                    c=peak_amplitude_df_now.magnitude, s=1, alpha=0.5, vmax=6, vmin=2, cmap=cmap)
    gca.set_ylim(-3, 2.5)
    gca.set_xlim(-1, 3)
    gca.set_xlabel('log10(hypo-distance)')
    gca.set_ylabel('log10(peak S)')
    gca.set_title(region_label)


plt.subplots_adjust(wspace=0.3, hspace=0.3)
# %%
