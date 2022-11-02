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
from scipy.signal import butter, filtfilt


import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

import warnings
# Add path 
import sys
sys.path.append("/home/yinjx/notebooks/strain_scaling")

from utility_functions import *
from obspy.geodetics import locations2degrees

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%
data_folder = '/kuafu/EventData/Sanriku_ERI/data'
h5_file_list = glob.glob(data_folder + '/*.h5')

h5_file = h5_file_list[1000]
event_id = h5_file[-7:-3]
event_data, event_info = load_event_data(data_folder, event_id)
# %%
# event_info['dx_m'] = 5

# f1 = h5py.File(h5_file, 'r')     # open the file
# data = f1['data']       # load the data
# data[...] = X1                      # assign new values to data
# f1.close()   

data_folder = '/kuafu/EventData/Sanriku_ERI/data'
h5_file_list = glob.glob(data_folder + '/*.h5')

for ii, h5_file in enumerate(h5_file_list):
    print(f'{ii/len(h5_file_list)*100}------' + h5_file)
    with h5py.File(h5_file, 'r+') as fid:
        fid['data'].attrs['dx_m'] = 5
# %%
