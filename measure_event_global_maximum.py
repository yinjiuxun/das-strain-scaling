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
event_folder_list = ['/kuafu/EventData/Ridgecrest', '/kuafu/EventData/Mammoth_north', '/kuafu/EventData/Mammoth_south', '/kuafu/EventData/Sanriku_ERI']
peak_amplitude_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events',
                           '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']

for ii_region in [3]:# [0, 1, 2]:
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

            if event_data.shape[1] >= das_info.shape[0]:
                event_data = event_data[:, das_info.index]
                array_maximum[i_eq, :] = np.nanmax(event_data, axis=0)
            else: # Sanriku data is slightly different, data has less channels than the DAS info (need to ask)
                array_maximum[i_eq, :event_data.shape[1]] = np.nanmax(event_data, axis=0)
        except:
            continue
    np.savez(peak_amplitude_dir + '/global_maximum.npz', array_maximum=array_maximum)

# %%
# Setup the paths
event_folder_list = ['/kuafu/EventData/Ridgecrest', '/kuafu/EventData/Mammoth_north', '/kuafu/EventData/Mammoth_south', '/kuafu/EventData/Sanriku_ERI']
peak_amplitude_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events',
                           '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']

site_term_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate/regression_results_smf_weighted/site_terms_10chan.csv', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/regression_results_smf_weighted/site_terms_10chan.csv', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/regression_results_smf_weighted/site_terms_10chan.csv',
                           '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/regression_results_smf_weighted_all_coefficients_drop_4130/site_terms_10chan.csv']

region_label = ['RC', 'LV-N', 'LV-S', 'Sanriku']

fig = plt.figure(figsize=(12, 16))
plt.subplots_adjust(wspace= 0.5, hspace= 0.3)
sub1 = fig.add_subplot(3,1,1) 
sub2 = fig.add_subplot(3,1,2) 
sub3 = fig.add_subplot(3,1,3) 

maximum_info = []

dx_list = [8, 10, 10, 5]
max_range = [110, 30, 30, 30]

ping_rate = np.array([5000, 2000, 1000, 500]) # Sanriku should be 500
gauge_length = np.array([16.34, 20.42, 20.42, 40.4])
clipping_factor = ping_rate/ gauge_length

array_clipping = []
for ii_region in [0, 1, 2, 3]:

    event_folder, peak_amplitude_dir = event_folder_list[ii_region], peak_amplitude_dir_list[ii_region]

    catalog = pd.read_csv(event_folder + '/catalog.csv')
    #ii_M2 = catalog.magnitude >=2

    temp = np.load(peak_amplitude_dir + '/global_maximum.npz')
    temp = temp['array_maximum']
    #temp = temp[ii_M2, :]

    # load site terms
    site_term = pd.read_csv(site_term_dir_list[ii_region])
    site_term_array = np.ones(temp.shape[1])*np.nan
    site_term_array[site_term.channel_id.values.astype('int')] = site_term.site_term_S
    # site_term_array = np.nanmedian(site_term_array)

    channel_max = temp.copy()
    channel_max[(channel_max >120) | (channel_max <=1e-1)] = np.nan
    channel_max = np.nanmax(channel_max, axis=0)


    global_max = temp[temp>0].flatten()

    array_max = np.amax(global_max[global_max<max_range[ii_region]])
    array_clipping.append(array_max)

    maximum_info.append(f'{region_label[ii_region]} :  log10({array_max}) = {np.log10(array_max)}\n')
    print(f'=================== {region_label[ii_region]} :  log10({array_max}) = {np.log10(array_max)} ========================== \n')

    gca = sub1
    temp = gca.hist(np.log10(global_max), range=(-4,8), bins=100, label=region_label[ii_region], alpha=0.5)
    gca.vlines(x=np.log10(array_max), ymin=1, ymax=1e8, linestyle='--', color=temp[2][1]._facecolor[:-1])
    gca.set_yscale('log')
    gca.set_ylim(ymin=1, ymax= 1e7)

    gca = sub2
    # gca.semilogy(np.linspace(0, 1, len(channel_max)), channel_max, label=region_label[ii_region])
    gca.semilogy(np.arange(len(channel_max)) * dx_list[ii_region]/1e3, channel_max, 
                '.', label=region_label[ii_region], alpha=0.5, zorder=-ii_region)

    gca = sub3
    # # remove the site terms
    # channel_max = channel_max/(10**site_term_array)
    # gca.semilogy(np.linspace(0, 1, len(channel_max)), channel_max, label=region_label[ii_region])
    gca.semilogy(np.arange(len(channel_max)) * dx_list[ii_region]/1e3, channel_max/clipping_factor[ii_region], 
                '.', label=region_label[ii_region], alpha=0.5, zorder=-ii_region)

sub1.annotate(f'(a)', xy=(-0.1, 0.95), xycoords=sub1.transAxes, fontsize=20)
sub1.grid()
sub1.legend(loc=1)
sub1.set_xlabel('log10(E) (micro strain/s)')
sub1.set_ylabel('Counts')

sub2.annotate(f'(b)', xy=(-0.1, 0.95), xycoords=sub2.transAxes, fontsize=20)
sub2.text(20, 70, 'OptSense Plexus')
sub2.text(50, 0.7, 'AP Sensing N5200A ')
sub2.set_ylim(1e-1, 2e2)
sub2.grid()
sub2.legend(loc=1)
sub2.set_xlabel('Distance to interrogator (km)')
sub2.set_ylabel('Clipping\n amplitude')

sub3.annotate(f'(c)', xy=(-0.1, 0.95), xycoords=sub3.transAxes, fontsize=20)
sub3.text(20, 0.7, 'OptSense Plexus')
sub3.text(50, 0.05, 'AP Sensing N5200A ')
sub3.set_ylim(1e-2, 1)
sub3.grid()
sub3.legend(loc=1)
sub3.set_xlabel('Distance to interrogator (km)')
sub3.set_ylabel('Scaled clipping\n amplitude')

with open('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/maximum_info.txt', 'w') as f:
    f.write(''.join(maximum_info))

plt.savefig('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/maximum_amplitude_distribution.pdf', bbox_inches='tight')
# %%


# %%
