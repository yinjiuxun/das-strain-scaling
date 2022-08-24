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


#%%
# Adding pyDAS location
pyDAS_path = "/home/yinjx/DAS-utilities/build/"
import os
try:
    os.environ['LD_LIBRARY_PATH'] += ":" + pyDAS_path
except:
    os.environ['LD_LIBRARY_PATH'] = pyDAS_path
import sys
sys.path.insert(0,pyDAS_path)
sys.path.insert(0,'/home/yinjx/DAS-utilities/python')
import DASutils
import importlib
importlib.reload(DASutils)

Ridgecrest_conversion_factor = 1550.12 / (0.78 * 4 * np.pi * 1.46 * 8)

# Save and load event hdf5 files
def save_rawevent_h5(fn, data, info):
   """
   """
   info_copy = info.copy()
   with h5py.File(fn, 'w') as fid:
       fid.create_dataset('data', data=data)
       for key in info.keys():
           if isinstance(info[key], str):
               #fid['data'].attrs.modify(key, np.string_(info_copy[key]))
               fid['data'].attrs.modify(key, info_copy[key])
           else:
               fid['data'].attrs.modify(key, info_copy[key])

def load_event_data(data_path, eq_id):
    with h5py.File(data_path + '/' + str(eq_id) + '.h5', 'r') as fid:
        data = fid['data'][:]
        info = {}
        for key in fid['data'].attrs.keys():
            info[key] = fid['data'].attrs[key]
        if 'begin_time' in info.keys():
            info['begin_time'] = dateutil.parser.parse(info['begin_time'])
        if 'end_time' in info.keys():
            info['end_time'] = dateutil.parser.parse(info['end_time'])
        if 'event_time' in info.keys():
            info['event_time'] = dateutil.parser.parse(info['event_time'])
    return data, info
# %%
DAS_info_df = pd.read_csv('/kuafu/EventData/Ridgecrest/das_info.csv')
#%%
data_dir = '/kuafu/DASdata/Ridgecrest_ODH3_2_Hourly/Ridgecrest_ODH3-2021-07-29 061539Z.h5'

data0, info = DASutils.readFile_HDF([data_dir], 0.5, 40.0, taper=0.05,filter=False,
                                        verbose=1, desampling=False, system="OptaSense", nChbuffer=5000)
data0 = data0[DAS_info_df['index'], :]
ntS = info['nt']
fs = info['fs']
das_dt = info['dt']
das_time = np.linspace(0.0, (ntS-1)*das_dt, ntS)

# convert to strain rate
data_diff = np.diff(data0, axis=1)/das_dt / 1e3 * Ridgecrest_conversion_factor # convert from nano strain rate to micro strain rate
# time information 
das_time = das_time[:-1] + das_dt/2

#%%
# event info
eq_time_UTCDateTime = [obspy.UTCDateTime('2021-07-29T061549.19')]
eq_das_t0 = [obspy.UTCDateTime('2021-07-29T061539')]
eq_id = ['0219neiszm']
eq_mag = [8.2]
# %%
i_event = 0
# Taper the data
t_begin = eq_time_UTCDateTime[i_event] - eq_das_t0[i_event] - 10
t_end = eq_time_UTCDateTime[i_event] - eq_das_t0[i_event] + 1000

# segmented data
ii_time = (das_time >= t_begin) & (das_time <= t_end)
data_diff = data_diff[:, ii_time]
das_time = das_time[ii_time]   

# Downsample from 250 Hz to 100 Hz using interp
das_dt_ds = 0.01
das_time_ds = np.arange(t_begin, t_end - das_dt_ds, das_dt_ds)

downsample_f = interp1d(das_time, data_diff, axis=1, bounds_error=False, fill_value=0)
data_diff_ds = downsample_f(das_time_ds)

event_data = data_diff_ds.T
event_info = {}
event_info['event_id'] = eq_id[i_event]
event_info['event_time'] = eq_time_UTCDateTime[i_event].isoformat()
event_info['begin_time'] = (eq_time_UTCDateTime[i_event] - 10).isoformat() + '+00:00'
event_info['end_time'] = (eq_time_UTCDateTime[i_event] + 1000).isoformat() + '+00:00'
event_info['latitude'] = 55.364
event_info['longitude'] = -157.888
event_info['depth_km'] = 35.0
event_info['magnitude'] = 8.2
event_info['magnitude_type'] = 'ms'
event_info['source'] = 'az'
event_info['dt_s'] = das_dt_ds
event_info['dx_m'] = 8
event_info['das_array'] = 'Ridgecrest'

#%%
data_folder = '/kuafu/EventData/Ridgecrest/data'
fig_folder = data_folder + '/figures'
save_rawevent_h5(os.path.join(data_folder, str(eq_id[i_event])+'.h5'), event_data, event_info)

#%%
# Show data
fig, ax1 = plt.subplots(figsize=(8,4))
pclip=80
clipVal = np.percentile(np.absolute(event_data), pclip)
# Vx
ax1.imshow(event_data, 
        extent=[0, event_data.shape[1], das_time[-1], das_time[0]],
        aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

# ax1.plot(np.arange(0, event_data.shape[1]), event_arrival_P[:, i_event], '--g', zorder=10)
# ax1.plot(np.arange(0, event_data.shape[1]), event_arrival_S[:, i_event], '-g', zorder=10)

ax1.set_xlabel("Channel number")
ax1.set_ylabel("Time [s]")
ax1.grid()
#ax1.set_ylim(200, 800)
#ax1.invert_yaxis()
ax1.set_title(f'Event id {eq_id[i_event]}, M {eq_mag[i_event]}')

plt.savefig(fig_folder + f'/{eq_id[i_event]}.png', bbox_inches='tight')
# ax1.set_title(f'Event id {eq_id[i_event]}, M {eq_mag[i_event]}')


# %%
# load the event data
eq_dir = '/kuafu/EventData/Ridgecrest/data'
eq_id = '0219neiszm'
event_data, event_info = load_event_data(eq_dir, eq_id)

#%%
das_time = np.arange(0, event_data.shape[0]) * event_info['dt_s']
plt.plot(das_time, event_data[:, 350])
plt.xlim(400, 800)

# %%
