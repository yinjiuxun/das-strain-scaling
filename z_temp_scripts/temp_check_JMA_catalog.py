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
from scipy.signal import butter, filtfilt

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


#%% Setup the paths
event_folder_list = ['/kuafu/EventData/Sanriku_ERI']
peak_amplitude_dir_list = ['/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']

event_folder, peak_amplitude_dir = event_folder_list[0], peak_amplitude_dir_list[0]

print('='*10 + peak_amplitude_dir + '='*10)

# Event waveform directory, phase picking directory
data_path = event_folder + '/data'
pick_path = event_folder + '/matched_filter_detections' # directory of ML phase picker

# Load the catalog, filter events, load waveforms
catalog = pd.read_csv(event_folder + '/catalog.csv')
das_info = pd.read_csv(event_folder + '/das_info.csv')
# %%
fig, gca = plt.subplots()
gca.scatter(das_info.longitude, das_info.latitude, s=10, c=das_info['index'], cmap='jet')
gca.scatter(catalog.longitude, catalog.latitude, s=2**catalog.magnitude, c='k', zorder=-1)
gca.set_xlim(np.mean(das_info.longitude)-2, np.mean(das_info.longitude)+2)
gca.set_ylim(np.mean(das_info.latitude)-2, np.mean(das_info.latitude)+2)
# %%
fig, gca = plt.subplots()
gca.hist(catalog.magnitude, range=(0, 7), bins=100)
# %%
