#%%
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

import sys
sys.path.append("../")
import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

from utility.general import *
#%%
def read_PASSCAL_SEGY_headers(infile):
    """Function to read information within SEGY PASSCAL headers"""
    nTxtFileHeader=3200
    nBinFileHeader=400
    nTraceHeader=240
    if infile.split(".")[-1] == "segy":
        fid = open(infile, 'rb')
    elif infile.split(".")[-1] == "gz":
        fid = gzip.open(infile, 'rb')
    fid.seek(0,2)
    filesize = fid.tell()
    startData = nTxtFileHeader+20
    fid.seek(startData)
    BinBuffer = fid.read(2)
    nt = int.from_bytes(BinBuffer, byteorder='big', signed=False)
    fid.seek(nTxtFileHeader+16)
    dt = int.from_bytes(fid.read(2), byteorder='big', signed=False)*1e-6
    if dt == 0.0:
        fs = 0.0
    else:
        fs = 1.0/dt
    # Getting UTC time first sample
    fid.seek(nTxtFileHeader+nBinFileHeader+156)
    year = int.from_bytes(fid.read(2), byteorder='big', signed=False)
    day = int.from_bytes(fid.read(2), byteorder='big', signed=False)
    hour = int.from_bytes(fid.read(2), byteorder='big', signed=False)
    minute = int.from_bytes(fid.read(2), byteorder='big', signed=False)
    second = int.from_bytes(fid.read(2), byteorder='big', signed=False)
    print([year, day, hour, minute, second])
    TIME_BASIS_CODE = int.from_bytes(fid.read(2), byteorder='big', signed=False)
    print(TIME_BASIS_CODE)
    micsec = int.from_bytes(fid.read(4), byteorder='big', signed=False)
    second = second+micsec*1e-6
    startTime = datetime.datetime.strptime("%s-%s"%(year,day),"%Y-%j") + datetime.timedelta(hours=hour, minutes=minute, seconds=second)
    print(startTime)
    if (TIME_BASIS_CODE) == 4 or (TIME_BASIS_CODE == 0):
        startTime = startTime.replace(tzinfo=pytz.utc)
    else:
        raise ValueError("Unknown time zone!")
    endTime = startTime + datetime.timedelta(seconds=nt*dt)
    nTraces = int((filesize - nTxtFileHeader - nBinFileHeader)/(nTraceHeader+nt*4))
    # Get the interregator info
    fid.seek(nTxtFileHeader+nBinFileHeader+232)
    ping_rate = int.from_bytes(fid.read(2), byteorder='big', signed=False)
    recv_depth = int.from_bytes(fid.read(4), byteorder='big', signed=False)
    gauge_length = int.from_bytes(fid.read(2), byteorder='big', signed=False)
    fid.close()
    return nt, fs, startTime, endTime, nTraces, (ping_rate, recv_depth, gauge_length)


def read_PASSCAL_segy(infile, nTraces, nSample, TraceOff=0):
    """Function to read PASSCAL segy raw data"""
    data = np.zeros((nTraces, nSample), dtype=np.float32)
    gzFile = False
    if infile.split(".")[-1] == "segy":
        fid = open(infile, 'rb')
    elif infile.split(".")[-1] == "gz":
        gzFile = True
        fid = gzip.open(infile, 'rb')
    fid.seek(3600)
    # Skipping traces if necessary
    fid.seek(TraceOff*(240+nSample*4),1)
    # Looping over traces
    for ii in range(nTraces):
        fid.seek(240, 1)
        if gzFile:
            # np.fromfile does not work on gzip file
            BinDataBuffer = fid.read(nSample*4) # read binary bytes from file
            data[ii, :] = struct.unpack_from(">"+('f')*nSample, BinDataBuffer)
        else:
            data[ii, :] = np.fromfile(fid, dtype=np.float32, count=nSample)
    fid.close()
    return data

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

def load_rawevent_h5(fn):
    # with h5py.File(fn, 'r') as fid:
    #    data = fid['data'][:]
    #    info = {}
    #    for key in fid['data'].attrs.keys():
    #        info[key] = fid['data'].attrs[key]
    #    info2 = {}
    #    if 'begin_time' in info.keys():
    #        info2['begTime'] = dateutil.parser.parse(info['begin_time'])
    #    if 'end_time' in info.keys():
    #        info2['endTime'] = dateutil.parser.parse(info['end_time'])
    #    if 'event_time' in info.keys():
    #        info2['time'] = dateutil.parser.parse(info['event_time'])
    #    info2['nt'] = data.shape[0]
    #    info2['nx'] = data.shape[1]
    #    info2['dx'] = info['dx_m']
    #    info2['dt'] = info['dt_s']
    

    with h5py.File(fn, 'r') as fid:
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
   

# This is used to convert the DAS phase shift to strain (in nano strain)
Ridgecrest_conversion_factor = 1550.12 / (0.78 * 4 * np.pi * 1.46 * 16.35)

#%%
# Write DAS info
DAS_info = np.genfromtxt('/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Ridgecrest_ODH3.txt')

DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info[:, 0].astype('int')
DAS_lon = DAS_info[:, 1]
DAS_lat = DAS_info[:, 2]

# Write to a csv in the EventData folder
DAS_info_df = pd.DataFrame({'index': DAS_index, 'latitude': DAS_lat, 'longitude': DAS_lon, 'elevation_m': DAS_info[:, 3]})
DAS_info_df.to_csv('/kuafu/EventData/Ridgecrest/das_info.csv', index=False)

#%% 
# load Ridgecrest catalog
catalog_file = '/home/yinjx/notebooks/strain_scaling/Ridgecrest_das_catalog_M0_M8.txt'
catalog = pd.read_csv(catalog_file, sep='\s+', header=None, skipfooter=1, engine='python')

catalog_select = catalog[(catalog[7] >= 1.5) & (catalog[6] > 1)] # choose the event with magnitude > 3.5
eq_num = catalog_select.shape[0]
eq_id = np.array(catalog_select[0])
eq_time = np.array(catalog_select[3])
eq_lat = np.array(catalog_select[4])
eq_lon = np.array(catalog_select[5])
eq_dep = np.array(catalog_select[6])
eq_mag = np.array(catalog_select[7])
eq_mag_type = 'm' + np.array(catalog_select[8])

print(f'Total number of events: {eq_num}')

# Convert the event time to obspy UTCDateTime and also find the corresponding DAS file name
import datetime
import obspy

# event time in obspy.UTCDateTime
eq_time_UTCDateTime = [obspy.UTCDateTime(temp) for temp in eq_time]
eq_time_str = [obspy.UTCDateTime(temp).isoformat() + '+00:00' for temp in eq_time]
# corresponding das file name
eq_das_files = [datetime.datetime.strptime(temp[:-4],'%Y/%m/%d,%H:%M:%S').strftime('%Y%m%d%H.segy') for temp in eq_time]
# corresponding das data initial time in UTCDateTime form
eq_das_t0 = [obspy.UTCDateTime(temp[0:-5]) for temp in eq_das_files]

# Write to a catalog.csv in the EventData folder
event_info_df = pd.DataFrame({'event_id': eq_id, 'event_time': eq_time_str,
'latitude': eq_lat, 'longitude': eq_lon, 'depth_km': eq_dep, 'magnitude': eq_mag, 'magnitude_type': eq_mag_type, 'source': 'scsn'})

event_info_df.to_csv('/kuafu/EventData/Ridgecrest/catalog.csv', index=False)

#%%
# Save the segmented data for ML phase picking
# This is to save as HDF5
from scipy.interpolate import interp1d

data_folder = '/kuafu/EventData/Ridgecrest/data'
fig_folder = data_folder + '/figures'

def save_segmented_event_data(i_event, DAS_index, eq_num, eq_id, eq_lat, eq_lon, eq_dep, eq_mag, eq_time_UTCDateTime, eq_das_files, eq_das_t0, data_folder, fig_folder):
    matplotlib.rcParams.update(params) # Set up the plotting parameters
    warnings.filterwarnings('ignore')

    # if os.path.exists(os.path.join(data_folder, str(eq_id[i_event])+'.h5')):
    #     print(f'Event data {eq_id[i_event]} exists, skip...')
    #     pass
    # else:
    
    if True:
        try:
    
            das_path = '/kuafu/zshen/Ridgecrest_data/1hoursegy/total/'
            file_name = eq_das_files[i_event]

            data0 = read_PASSCAL_segy(das_path + file_name, 1250, 900000, 0)
            data0 = data0[DAS_index, :]


            das_dt = 3600 / data0.shape[1]
            data_diff = np.diff(data0, axis=1)/das_dt / 1e3 * Ridgecrest_conversion_factor # convert from nano strain rate to micro strain rate

            # time information 
            das_time = np.linspace(das_dt/2,3600-das_dt/2,data_diff.shape[1])

            # Taper the data
            t_begin = eq_time_UTCDateTime[i_event] - eq_das_t0[i_event] - 30
            t_end = eq_time_UTCDateTime[i_event] - eq_das_t0[i_event] + 90

            # segmented data
            ii_time = (das_time >= t_begin) & (das_time <= t_end)
            data_diff = data_diff[:, ii_time]
            das_time = das_time[ii_time]   

            # Downsample from 250 Hz to 100 Hz using interp
            das_dt_ds = 0.01
            das_time_ds = np.linspace(t_begin, t_end - das_dt_ds, 12000)

            downsample_f = interp1d(das_time, data_diff, axis=1, bounds_error=False, fill_value=0)
            data_diff_ds = downsample_f(das_time_ds)

            event_data = data_diff_ds.T
            event_info = {}
            event_info['event_id'] = eq_id[i_event]
            event_info['event_time'] = eq_time_UTCDateTime[i_event].isoformat()
            event_info['begin_time'] = (eq_time_UTCDateTime[i_event] - 30).isoformat() + '+00:00'
            event_info['end_time'] = (eq_time_UTCDateTime[i_event] + 90).isoformat() + '+00:00'
            event_info['latitude'] = eq_lat[i_event]
            event_info['longitude'] = eq_lon[i_event] 
            event_info['depth_km'] = eq_dep[i_event]
            event_info['magnitude'] = eq_mag[i_event]
            event_info['magnitude_type'] = 'ml'
            event_info['source'] = 'scsn'
            event_info['dt_s'] = das_dt_ds
            event_info['dx_m'] = 8
            event_info['das_array'] = 'Ridgecrest'

            save_rawevent_h5(os.path.join(data_folder, str(eq_id[i_event])+'.h5'), event_data, event_info)


            # Show data
            fig, ax1 = plt.subplots(figsize=(8,4))
            pclip=99.5
            clipVal = np.percentile(np.absolute(data_diff), pclip)
            # Vx
            ax1.imshow(event_data, 
                    extent=[0, event_data.shape[1], das_time[-1], das_time[0]],
                    aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

            # ax1.plot(np.arange(0, event_data.shape[1]), event_arrival_P[:, i_event], '--g', zorder=10)
            # ax1.plot(np.arange(0, event_data.shape[1]), event_arrival_S[:, i_event], '-g', zorder=10)

            ax1.set_xlabel("Channel number")
            ax1.set_ylabel("Time [s]")
            ax1.grid()
            ax1.set_title(f'Event id {eq_id[i_event]}, M {eq_mag[i_event]}')

            plt.savefig(fig_folder + f'/{eq_id[i_event]}.png', bbox_inches='tight')
            plt.close('all')

        except:
            pass

# i_event=100
# save_segmented_event_data(i_event, DAS_index, eq_num, eq_id, eq_lat, eq_lon, eq_dep, eq_mag, 
# eq_time_UTCDateTime, eq_das_files, eq_das_t0, data_folder, fig_folder)
# %%
# n_run = 1000
n_run = eq_num
with tqdm_joblib(tqdm(desc="extracting segmented event data", total=n_run)) as progress_bar:
    Parallel(n_jobs=Ncores)(delayed(save_segmented_event_data)(i_event, DAS_index, eq_num, eq_id, eq_lat, eq_lon, eq_dep, eq_mag, 
                            eq_time_UTCDateTime, eq_das_files, eq_das_t0, data_folder, fig_folder) for i_event in range(n_run))

# %%
