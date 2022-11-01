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

from ..utility_functions import *
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

# %% 
# functions
from obspy import UTCDateTime as UTC
def read_prem_file(file):
    
    #### Return detection time, magntiude, depth, distance and a list of matched channels of the earthquake
    #### Note other channels may also detect the earthquake, but the similarity doesn't pass certain threshold
    #### Also note hand picked files don't have matched channels info

    f=open(file,'r')
    sampling_rate=100
    
    lines=f.readlines()
    for line in lines:
        if line.startswith('File start time'):
            stime=UTC(line.split('File start time (UTC+9):')[1])
            break
            
    dtime=stime+20 # detection time
    etime=stime+60 # end time
    
    for line in lines:
        if line.startswith('Magnitude'):
            mag=line.split('Magnitude: ')[1]
            
            if mag == 'unknown':
                pass
            else:
                mag=float(mag)
            break
                
    for line in lines:
        if line.startswith('Depth'):
            dep=line.split('Depth (km): ')[1]
            
            if dep == 'unknown':
                pass
            else:
                dep=float(dep)
            break
            
    for line in lines:
        if line.startswith('Distance'):
            dist=line.split('Distance (km): ')[1]
            
            if dist == 'unknown':
                pass
            else:
                dist=float(dist)
            break

    for line in lines:
        if line.startswith('Passed QA'):
            QA = line.split('Passed QA text: ')[1]
            QA = QA[:-1] # remove /n
            break
            
    channels=[]
    for line in lines:
        if line.startswith('Matched channels'):
            idx=lines.index(line)
            
            for i in range(idx+1,len(lines)):
                channels.append(int(lines[i]))
                
            break
                
    return sampling_rate, stime, dtime, etime, mag, dep, dist, QA, channels


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

# plot waveform
def plotting_waveform(event_data, clipVal, detect_channels, gca):
    gca.imshow(event_data, 
        extent=[0, event_data.shape[1], event_data.shape[0]/100, 0],
        aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

    gca.plot(detect_channels, 20*np.ones(len(detect_channels)), 'g.', linewidth=4, markersize=5, alpha=0.6)

    gca.set_xlim(xmax=np.max(detect_channels))
    gca.set_xlabel('Channels')
    gca.set_ylabel('Time (s)')

    return gca

# %% Reorganize the JMA catalog
catalog_raw = pd.read_csv('/kuafu/yinjx/Japan_DAS/JMA.csv')
catalog = pd.DataFrame(columns=['event_id','event_time', 'latitude', 'longitude', 'depth_km','magnitude','magnitude_type','source'
])

# Adjust the Event time
temp_time = catalog_raw['Date'].astype('str') + 'T' + catalog_raw['Time'].astype('str')
eq_time_str = [(obspy.UTCDateTime(temp)-9*3600).isoformat() + '+00:00' for temp in temp_time] # JMA is in Japan time, convert to UTC

# Adjust the magnitude
temp_magnitude = catalog_raw['Magnitude'].astype('str')
ii_MD = temp_magnitude.str.contains('D', case=False) # displacement magnitude
ii_MV = temp_magnitude.str.contains('V', case=False) # velocity magnitude
ii_MW = temp_magnitude.str.contains('W', case=False) # moment magnitude
eq_magnitude = np.array([temp[:3]for temp in temp_magnitude]).astype('float')

# Fill in the information
catalog['event_id'] = catalog_raw['Index']
catalog['event_time'] = eq_time_str
catalog['latitude'] = catalog_raw['Latitude']
catalog['longitude'] = catalog_raw['Longitude']
catalog['depth_km'] = catalog_raw['Depth']
catalog['magnitude'] = eq_magnitude

catalog['magnitude_type'][ii_MD] = 'Md'
catalog['magnitude_type'][ii_MV] = 'Mv'
catalog['magnitude_type'][ii_MW] = 'Mw'

catalog['source'] = 'JMA'

catalog = catalog.dropna()
catalog.to_csv('/kuafu/EventData/Sanriku_ERI/catalog.csv', index=False)

# %% Reorganize the Japan DAS info file
das_info = pd.read_csv('/kuafu/yinjx/Japan_DAS/das_info.csv')
das_info['elevation_m'] = np.nan
das_info.to_csv('/kuafu/EventData/Sanriku_ERI/das_info.csv', index=False)


# %% Reorganize the Japan DAS data
def reorganize_event_waveform(catalog, data_folder, i_event):
    matplotlib.rcParams.update(params) # Set up the plotting parameters
    warnings.filterwarnings('ignore')

    fig_folder = data_folder + '/figures'
    try:
        eq_id = catalog.event_id[i_event]

        # event data info
        event_info_file = f'/kuafu/yinjx/Japan_DAS/prem_model_events/index{eq_id}/index{eq_id}_info.txt'
        # event data
        event_data_file = f'/kuafu/yinjx/Japan_DAS/prem_model_events/index{eq_id}/index{eq_id}.h5'


        # load the event info
        sampling_rate, begin_time, detect_time, end_time, magnitude, depth_km, epi_distance, QA, detect_channels = read_prem_file(event_info_file)

        event_info = {}
        event_info['event_id'] = eq_id
        event_info['event_time'] = catalog.event_time[i_event]
        event_info['begin_time'] = begin_time.isoformat() + '+00:00'
        event_info['end_time'] = end_time.isoformat() + '+00:00'
        event_info['latitude'] = catalog.latitude[i_event]
        event_info['longitude'] = catalog.longitude[i_event]
        event_info['depth_km'] = catalog.depth_km[i_event]
        event_info['magnitude'] = catalog.magnitude[i_event]
        event_info['magnitude_type'] = catalog.magnitude_type[i_event]
        event_info['source'] = catalog.source[i_event]
        event_info['dt_s'] = 1/sampling_rate
        event_info['dx_m'] = 5
        event_info['unit'] = 'microstrain/s'
        event_info['das_array'] = 'sanriku'
        event_info['pass_QA'] = QA

        # convert the phase to strain rate (micro strain rate)
        sanriku_conversion_factor = np.pi/32768*1550*1e-9/(4*np.pi*1.468*40*0.78) * 1e6 # convert phase to microstrain

        # load the event data
        with h5py.File(event_data_file, 'r') as f:
                event_data = f['DAS'][:]
        event_data = event_data.T # reshape to n_time * n_channel
        
        event_data = np.cumsum(event_data, axis=0) * sanriku_conversion_factor # integrate to phase, then convert to strain
        event_data = np.gradient(event_data, 1/sampling_rate, axis=0) # to strain rate
        

        save_rawevent_h5(os.path.join(data_folder, str(eq_id)+'.h5'), event_data, event_info)

        # %% save the detection from matched filter
        MF_folder = '/kuafu/EventData/Sanriku_ERI/matched_filter_detections'
        np.savez(MF_folder + f'/{eq_id}_MF_detect.npz', detect_time=detect_time, detect_channels=detect_channels, QA=QA, allow_pickle=True)

        # test_detect = np.load(MF_folder + f'/{eq_id}_MF_detect.npz', allow_pickle=True)
        # test_detect['detect_time']
        # test_detect['detect_channels']


        # apply high pass filter
        dt_s = 1 / sampling_rate
        f_pass = 0.5
        aa, bb = butter(4, f_pass*2*dt_s, 'high')
        event_data_filter = filtfilt(aa, bb, event_data, axis=0)


        # %% plot a figure for data visualization (No filter)
        # show waveforms
        fig, ax = plt.subplots(2, 1, figsize=(8,10), sharex=True, sharey=True)
        ax = ax.flatten()

        pclip=90
        clipVal = np.percentile(np.absolute(event_data), pclip)

        gca = ax[0]
        gca = plotting_waveform(event_data, clipVal, detect_channels, gca)
        gca.set_title(f'ID: {eq_id}, {catalog.magnitude_type[i_event]} {catalog.magnitude[i_event]}, Pass QA: {QA}')

        gca = ax[1]
        gca = plotting_waveform(event_data_filter, clipVal, detect_channels, gca)
        gca.set_title(f'ID: {eq_id}, {catalog.magnitude_type[i_event]} {catalog.magnitude[i_event]}, Pass QA: {QA}, Highpass {f_pass} Hz')

        plt.figure(fig.number)
        plt.savefig(fig_folder + f'/{eq_id}.png', bbox_inches='tight')
        plt.close('all')
    
    except:
        print('nothing')
        pass

# %%
data_folder = '/kuafu/EventData/Sanriku_ERI/data'

i_event = 3187
reorganize_event_waveform(catalog, data_folder, i_event)

n_eq = catalog.shape[0]
with tqdm_joblib(tqdm(desc="File desampling", total=n_eq)) as progress_bar:
    Parallel(n_jobs=Ncores)(delayed(reorganize_event_waveform)(catalog, data_folder, i_event) for i_event in range(n_eq))

