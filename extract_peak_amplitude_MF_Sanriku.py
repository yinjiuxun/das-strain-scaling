#%%
# Import modules
import pandas as pd
#from sep_util import read_file

import numpy as np

import tqdm

import glob
import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed
from scipy.signal import butter, filtfilt

import warnings

import obspy
from obspy.geodetics import locations2degrees

# Plotting
import matplotlib
import matplotlib.pyplot as plt
# load modules
from utility.processing import calculate_SNR
from utility.loading import load_event_data, load_phasenet_pick
from utility.general import *
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

#%% 
# Functions used here
def apply_highpass_filter(event_data, f_pass, dt):
    aa, bb = butter(4, f_pass*2*dt, 'high')
    event_data_filter = filtfilt(aa, bb, event_data, axis=0)
    return event_data_filter

def show_event_data(event_data, das_time, gca, pclip=99.5):
    if pclip is None:
        clipVal = np.amax(abs(event_data))/20
    else:
        clipVal = np.percentile(np.absolute(event_data), pclip)
    gca.imshow(event_data, 
            extent=[0, event_data.shape[1], das_time[-1], das_time[0]],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

    gca.set_xlabel("Channel number")
    gca.set_ylabel("Time [s]")
    gca.grid()
    return gca

def extract_maximum_amplitude(time_list, data_matrix, t1, t2):
    '''Function used to extract the maximum within time window t1 and t2'''
    t1 = t1[np.newaxis, :]
    t2 = t2[np.newaxis, :]
    time_list = time_list[:, np.newaxis]

    # broadcast comparison
    mask_index = (time_list >= t1) & (time_list < t2)
    data_matrix_mask = data_matrix.copy()
    # first extract the time of maximum P
    data_matrix_mask[~mask_index] = 0
    data_matrix_mask = abs(data_matrix_mask)

    max_index = np.nanargmax(data_matrix_mask, axis=0)

    max_time = time_list[max_index, :]
    max_time[max_index==0, :] = np.nan
    max_time[max_index==(time_list.shape[0]-1), :] = np.nan
    max_time_ref1 = max_time.flatten() - t1.flatten()
    max_time_ref2 = t2.flatten() - max_time.flatten() 
    # then extract the maximum P 
    data_matrix_mask[~mask_index] = np.nan
    max_amplitude = np.nanmax(data_matrix_mask, axis=0)
    
    return max_amplitude, max_time_ref1, max_time_ref2

# %%
def extract_peak_amplitude_Sanriku(peak_amplitude_dir, data_path, pick_path, catalog, das_info, i_eq):
    matplotlib.rcParams.update(params) # Set up the plotting parameters
    warnings.filterwarnings('ignore')

    try:
        event_now = catalog.iloc[i_eq, :]
        event_data, event_info = load_event_data(data_path, event_now.event_id)
        event_data0 = event_data.copy()

        if event_data.shape[1] > das_info.shape[0]:
            event_data = event_data[:, das_info.index]

        nt = event_data.shape[0]
        das_time = np.arange(0, nt) * event_info['dt_s']

        #%% For the Sanriku case, the event time is based on the matched filter detection
        temp_npz = np.load(pick_path + f'/{event_now.event_id}_MF_detect.npz', allow_pickle=True)

        detect_time = temp_npz['detect_time']
        detect_channels = temp_npz['detect_channels']
        QA = temp_npz['QA']

        detect_time = detect_time - obspy.UTCDateTime(event_info['begin_time'])

        #%% Only focus on detected channels
        das_info = das_info.iloc[detect_channels, :]
        event_data = event_data[:, detect_channels]

        # apply highpass filter 0.5 Hz to remove ocean noise
        event_data = apply_highpass_filter(event_data, 0.5, event_info['dt_s'])

        #%% For the Sanriku data, since the MF detections are dominated by S wave, so only use the detect time as S picks
        pick_P, channel_P = np.nan * np.ones(detect_channels.shape), detect_channels
        pick_S, channel_S = detect_time * np.ones(detect_channels.shape), detect_channels


        # Calculate the SNR based on the P arrival time
        event_arrival_P_temp = pick_P[np.newaxis, :]
        event_arrival_S_temp = pick_S[np.newaxis, :]

        twin_noise_S = [np.ones(detect_channels.shape)*0, np.ones(detect_channels.shape)*10]
        twin_signal_S = [event_arrival_S_temp-10, event_arrival_S_temp + 10]
        snrP = np.nan * np.ones(detect_channels.shape)

        snrS = calculate_SNR(das_time, event_data, twin_noise_S, twin_signal_S)


        #%% Extract peak amplitude 
        # parameters of time window
        P_window_list = [2, 1, 3, 4] # 100 here mean S_pick-0.5
        S_window_list = [10, 8, 6, 4, 2]


        # Extract the maximum given the P and S arrival time
        max_S_amplitude_list = []
        max_S_time_list = []


        for S_window in S_window_list:
            max_S_amplitude, max_S_time, _ = extract_maximum_amplitude(das_time, event_data, pick_S-S_window, pick_S+S_window)
            max_S_amplitude_list.append(max_S_amplitude)
            max_S_time_list.append(max_S_time)
            if S_window == S_window_list[0]:
                peak_S_time = max_S_time.copy()


        # Convert to array
        max_P_amplitude_list = np.ones((len(detect_channels), len(P_window_list))) * np.nan
        max_P_time_list = np.ones((len(detect_channels), len(P_window_list))) * np.nan
        max_S_amplitude_list = np.array(max_S_amplitude_list).T
        max_S_time_list = np.array(max_S_time_list).T

        # Source information from catalog
        event_id_list = (np.ones(das_info.shape[0]) * event_now.event_id).astype('int')
        distance_list = locations2degrees(das_info.latitude, das_info.longitude, event_now.latitude, event_now.longitude) * 113
        magnitude_list = np.ones(das_info.shape[0]) * event_now.magnitude
        channel_id_list = np.array(das_info.index)
        # combined into one 
        all_combined1 = np.array([event_id_list, magnitude_list, channel_id_list, distance_list, snrP, snrS]).T

        all_combined = np.concatenate([all_combined1, 
                                max_P_amplitude_list, max_P_time_list, max_S_amplitude_list, max_S_time_list], axis=1)

        # Convert to DataFrame
        peak_amplitude_df = pd.DataFrame(data=all_combined, 
                                    columns=['event_id', 'magnitude', 'channel_id', 'distance_in_km', 'snrP', 'snrS', 
                                    'peak_P', 'peak_P_1s', 'peak_P_3s', 'peak_P_4s', 
                                    'peak_P_time', 'peak_P_time_1s', 'peak_P_time_3s', 'peak_P_time_4s',
                                    'peak_S', 'peak_S_8s', 'peak_S_6s', 'peak_S_4s', 'peak_S_2s', 
                                    'peak_S_time', 'peak_S_time_8s', 'peak_S_time_6s', 'peak_S_time_4s', 'peak_S_time_2s'])

        # # Remove NaNs
        # peak_amplitude_df = peak_amplitude_df.dropna()
        # Adjust the format
        peak_amplitude_df.event_id = peak_amplitude_df.event_id.astype('int')
        # Store to csv
        peak_amplitude_df.to_csv(peak_amplitude_dir + f'/{event_now.event_id}.csv',index=False)

        # Show data
        event_data0 = apply_highpass_filter(event_data0, 0.5, event_info['dt_s'])

        fig, ax = plt.subplots(figsize=(10,5))
        # Main
        gca = ax
        show_event_data(event_data0, das_time, gca, pclip=95)
        cbar = gca.scatter(channel_S, pick_S + peak_S_time - 10, s=10, c=snrS, cmap='jet')
        plt.colorbar(cbar, label='SNR')
        gca.plot(channel_S, pick_S, '.k', markersize=2)


        gca.set_title(f'ID {event_now.event_id}, M {event_now.magnitude}')
        plt.savefig(peak_amplitude_dir + f'/figures/{event_now.event_id}.png', bbox_inches='tight')
        plt.close('all')
    except:
        print('nothing')
        pass

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

# i_eq = 1550
# extract_peak_amplitude_Sanriku(peak_amplitude_dir, data_path, pick_path, catalog, das_info, i_eq)

n_eq = catalog.shape[0]
with tqdm_joblib(tqdm(desc="extracting peaks", total=n_eq)) as progress_bar:
    Parallel(n_jobs=Ncores)(delayed(extract_peak_amplitude_Sanriku)(peak_amplitude_dir, data_path, pick_path, catalog, das_info, i_eq) for i_eq in range(n_eq))


# %%
# Combine all the individual peak amplitude files into one for regression
event_folder_list = ['/kuafu/EventData/Sanriku_ERI']
peak_amplitude_dir_list = ['/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']

for ii_region in [0]:
    peak_amplitude_dir = peak_amplitude_dir_list[ii_region]

    print('='*10 + peak_amplitude_dir + '='*10)
    temp = glob.glob(peak_amplitude_dir + '/*.csv')
    temp_df = pd.concat(map(pd.read_csv, temp), ignore_index=True)
    temp_df.to_csv(peak_amplitude_dir + '/peak_amplitude.csv', index=False)

# %%
# Attach the QA factor
peak_amplitude_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events'
peak_amplitude_df = pd.read_csv(peak_amplitude_dir + '/peak_amplitude.csv')
peak_amplitude_df['QA'] = np.nan

event_id_list = peak_amplitude_df.event_id.unique()
pick_path = '/kuafu/EventData/Sanriku_ERI/matched_filter_detections'
for event_id in event_id_list:
    temp_npz = np.load(pick_path + f'/{event_id}_MF_detect.npz', allow_pickle=True)
    QA = temp_npz['QA']

    peak_amplitude_df.loc[peak_amplitude_df.event_id == event_id, 'QA'] = QA

peak_amplitude_df.to_csv(peak_amplitude_dir + '/peak_amplitude.csv', index=False)