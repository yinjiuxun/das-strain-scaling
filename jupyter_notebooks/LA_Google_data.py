#%% load module
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
import glob
import pytz
import warnings

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

import sys
sys.path.append("../")
from utility_functions import *
import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

from scipy.interpolate import interp1d

#%% Adding pyDAS location
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
# %%
# Prepare the catalog file
catalog_file = '/kuafu/EventData/LA_Google/SCEDC_catalog.txt'
catalog = pd.read_csv(catalog_file, sep='\s+', header=0, skipfooter=1, engine='python')

catalog_select = catalog# choose the event with magnitude > 3.5
eq_num = catalog_select.shape[0]
eq_id = np.array(catalog_select['EVID'])
eq_time = np.array(catalog_select['YYY/MM/DD'].astype('string')) + 'T' + np.array(catalog_select['HH:mm:SS.ss'].astype('string'))
eq_lat = np.array(catalog_select['LAT'])
eq_lon = np.array(catalog_select['LON'])
eq_dep = np.array(catalog_select['DEPTH'])
eq_mag = np.array(catalog_select['MAG'])
eq_mag_type = 'm' + np.array(catalog_select['M'])

print(f'Total number of events: {eq_num}')

# Convert the event time to obspy UTCDateTime and also find the corresponding DAS file name
# event time in obspy.UTCDateTime
eq_time_UTCDateTime = [obspy.UTCDateTime(temp) for temp in eq_time]
eq_time_str = [obspy.UTCDateTime(temp).isoformat() + '+00:00' for temp in eq_time]

# Write to a catalog.csv in the EventData folder
event_info_df = pd.DataFrame({'event_id': eq_id, 'event_time': eq_time_str,
'latitude': eq_lat, 'longitude': eq_lon, 'depth_km': eq_dep, 'magnitude': eq_mag, 'magnitude_type': eq_mag_type, 'source': 'scsn'})

event_info_df.to_csv('/kuafu/EventData/LA_Google/catalog.csv', index=False)

#%%
# DAS location "/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS-LAX_coor_interp.csv"
# Write DAS info
DAS_info = pd.read_csv('/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS-LAX_coor_interp.csv')

# 4780 channels in the data file, but 4790 in the DAS info...Calibrate that...
DAS_channel_num = 4780

# The incorrect lat lon files...
points = np.array([DAS_info.Longitude,
                   DAS_info.Latitude]).T  # a (nbre_points x nbre_dim) array

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

# Interpolation with slinear:
interpolations_method = 'slinear'
alpha = np.linspace(0, 1, DAS_channel_num)

interpolator =  interp1d(distance, points, kind=interpolations_method, axis=0)
interpolated_points = interpolator(alpha)

DAS_index = np.arange(0, DAS_channel_num)
DAS_lon = interpolated_points[:, 0]
DAS_lat = interpolated_points[:, 1]

# Write to a csv in the EventData folder
DAS_info_df = pd.DataFrame({'index': DAS_index, 'latitude': DAS_lat, 'longitude': DAS_lon, 'elevation_m': 0.0, 'quality': 'good'})

# label the bad channels
bad_channel_index = np.concatenate([np.arange(0, 201), np.arange(710, 811), np.arange(1160, 1361), np.arange(2360, 2661), np.arange(4730, 4780)])
DAS_info_df.quality[bad_channel_index] = 'bad'

DAS_info_df.to_csv('/kuafu/EventData/LA_Google/das_info_all.csv', index=False)

DAS_info_df = DAS_info_df[DAS_info_df.quality == 'good']
DAS_info_df.to_csv('/kuafu/EventData/LA_Google/das_info.csv', index=False)
# %%
# prepare the data file 
data_dir = '/kuafu/DASdata/GoogleData/LAX36-LAX28/'
data_files = glob.glob(data_dir + '*.h5')

data_folder = '/kuafu/EventData/LA_Google/data'
fig_folder = data_folder + '/figures'
# %%
# save DAS events
def save_das_event_data(event_now, DAS_info_df, data_files, data_folder, fig_folder):
    """
    Function to save event data in parallel
    """
    matplotlib.rcParams.update(params) # Set up the plotting parameters
    warnings.filterwarnings('ignore')

    event_time0  = obspy.UTCDateTime(event_now.event_time)

    data_files_time_t_begin = np.array([obspy.UTCDateTime(data_files[i][38:57]) for i in range(len(data_files))])
    data_files_time_t_end = np.array([obspy.UTCDateTime(data_files[i][58:77]) for i in range(len(data_files))])

    temp = np.sign(data_files_time_t_begin - event_time0) * np.sign(data_files_time_t_end - event_time0)
    i_file = np.argmin(temp)
    if temp[i_file] < 0: # find the data file, now extract the event data
        data_file = data_files[i_file]

        with h5py.File(data_file,'r') as fid:
            
            G = fid["Data"].attrs['GaugeLength']
            lam = fid["Data"].attrs['LaserWavelength']
            n = fid["Data"].attrs['RefractiveIndex']
            ChSamp = fid['Data'].attrs['ChSamp']
            conversion_factor = lam / (0.78 * 4 * np.pi * n * G) *1e6 *np.pi/2**15 # convert to micro strain

            data0 = fid["Data"][:] * conversion_factor
            ntS = fid["Data"].attrs['nt']
            fs = fid["Data"].attrs['fs']
            das_dt = fid["Data"].attrs['dt']

        das_time0 = np.linspace(0.0, (ntS-1)*das_dt, ntS)
        data_diff = np.gradient(data0, axis=1)/das_dt # to strain rate

        # Taper the data
        t_begin = event_time0 - data_files_time_t_begin[i_file] - 30
        t_end = event_time0 - data_files_time_t_begin[i_file] + 90

        # Because the data is only 50 Hz ... I have to oversample to get 100 Hz...
        das_dt = das_dt/2
        das_time = np.arange(t_begin, t_end-das_dt, das_dt)


        resample_f = interp1d(das_time0, data_diff, axis=1, bounds_error=False, fill_value=0)
        data_diff_rs = resample_f(das_time)
        fs = fs*2

        # %
        event_data = data_diff_rs.T
        # print(event_data.shape)
        # print(np.percentile(np.absolute(event_data), 80))
        event_data = event_data[:, DAS_info_df['index']]
        event_info = {}
        event_info['event_id'] = event_now.event_id
        event_info['event_time'] = event_now.event_time
        event_info['begin_time'] = (event_time0 - 30).isoformat() + '+00:00'
        event_info['end_time'] = (event_time0 + 90).isoformat() + '+00:00'
        event_info['latitude'] = event_now.latitude
        event_info['longitude'] = event_now.longitude
        event_info['depth_km'] = event_now.depth_km
        event_info['magnitude'] = event_now.magnitude
        event_info['magnitude_type'] = event_now.magnitude_type
        event_info['source'] = 'scsn'
        event_info['dt_s'] = das_dt
        event_info['dx_m'] = ChSamp
        event_info['das_array'] = 'LAX'

        save_rawevent_h5(os.path.join(data_folder, str(event_now.event_id)+'.h5'), event_data, event_info)
        # %

        # Show data
        fig, ax1 = plt.subplots(figsize=(8,4))
        pclip=70
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
        ax1.set_title(f'Event id {event_now.event_id}, M {event_now.magnitude}')

        plt.savefig(fig_folder + f'/{event_now.event_id}.png', bbox_inches='tight')
        plt.close('all')

    else:
        pass


event_info_df = event_info_df[event_info_df.magnitude >=1.5]

# save_das_event_data(event_info_df.iloc[0, ], DAS_info_df, data_files, data_folder, fig_folder)

n_run = event_info_df.shape[0]
with tqdm_joblib(tqdm(desc="extracting segmented event data", total=n_run)) as progress_bar:
    Parallel(n_jobs=Ncores)(delayed(save_das_event_data)(event_info_df.iloc[i_event, ], DAS_info_df, data_files, data_folder, fig_folder) for i_event in range(n_run))

# %%
