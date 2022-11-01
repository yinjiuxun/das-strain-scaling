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

import sys 

sys.path.append("../")

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

def extract_peak_amplitude_Sanriku(data_path, pick_path, catalog, das_info, eq_id, fig_path, peak_path):
    matplotlib.rcParams.update(params) # Set up the plotting parameters
    warnings.filterwarnings('ignore')

    try:
        event_now = catalog[catalog.event_id == eq_id]
        event_now = event_now.iloc[0, :]
        event_data, event_info = load_event_data(data_path, event_now.event_id)
        event_data0 = event_data.copy()

        if event_data.shape[1] > das_info.shape[0]:
            event_data = event_data[:, das_info.index]

        nt = event_data.shape[0]
        das_time = np.arange(0, nt) * event_info['dt_s']

        #% For the Sanriku case, the event time is based on the matched filter detection
        temp_npz = np.load(pick_path + f'/{event_now.event_id}_MF_detect.npz', allow_pickle=True)

        detect_time = temp_npz['detect_time']
        detect_channels = temp_npz['detect_channels']
        QA = temp_npz['QA']

        detect_time = detect_time - obspy.UTCDateTime(event_info['begin_time'])

        #% Only focus on detected channels
        das_info = das_info.iloc[detect_channels, :]
        event_data = event_data[:, detect_channels]

        # apply highpass filter 0.5 Hz to remove ocean noise
        event_data = apply_highpass_filter(event_data, 0.5, event_info['dt_s'])

        #% For the Sanriku data, since the MF detections are dominated by S wave, so only use the detect time as S picks
        pick_P, channel_P = np.nan * np.ones(detect_channels.shape), detect_channels
        pick_S, channel_S = detect_time * np.ones(detect_channels.shape), detect_channels

        # Show data
        event_data0 = apply_highpass_filter(event_data0, 0.5, event_info['dt_s'])

        # load peak amplitude 
        peak_amplitude_df = pd.read_csv(peak_path + '/calibrated_peak_amplitude.csv')
        mean_dist_in_km = peak_amplitude_df[peak_amplitude_df.event_id == eq_id].distance_in_km.mean()

        fig, ax = plt.subplots(figsize=(10,5))
        # Main
        gca = ax
        show_event_data(event_data0, das_time, gca, pclip=95)
        # cbar = gca.scatter(channel_S, pick_S + peak_S_time - 10, s=10, c=snrS, cmap='jet')
        # plt.colorbar(cbar, label='SNR')
        gca.plot(channel_S, pick_S, '.k', markersize=2)


        gca.set_title(f'ID {eq_id}, M {event_now.magnitude}, D {mean_dist_in_km:.2f} km')
        fig.savefig(fig_path + f'{eq_id}_fig.png', bbox_inches='tight')
        plt.close('all')
        return gca
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
fig_path = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/special_events/'
peak_path = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events'
# Load the catalog, filter events, load waveforms
catalog = pd.read_csv(event_folder + '/catalog.csv')
das_info = pd.read_csv(event_folder + '/das_info.csv')

event_to_look = pd.read_csv(fig_path + '/events_below_2.csv')
eq_id_list = np.array(event_to_look.event_id)
eq_id = 4479#3422
gca = extract_peak_amplitude_Sanriku(data_path, pick_path, catalog, das_info, eq_id, fig_path, peak_path)

# %%
n_eq = len(eq_id_list)
with tqdm_joblib(tqdm(desc="check events", total=n_eq)) as progress_bar:
    Parallel(n_jobs=Ncores)(delayed(extract_peak_amplitude_Sanriku)(data_path, pick_path, catalog, das_info, eq_id_list[i_eq], fig_path, peak_path) for i_eq in range(n_eq))


# %%
# to check a few examples
event_list_to_compare = np.arange(3280, 3286)
catalog_temp = catalog[catalog.event_id.isin(event_list_to_compare)]

plt.plot(das_info.longitude, das_info.latitude, 'k.')
plt.plot(catalog_temp.longitude, catalog_temp.latitude, 'r.')
for ii in range(len(event_list_to_compare)):
    # event_now = catalog_temp[catalog_temp.event_id == event_list_to_compare[ii]]
    plt.text(catalog_temp.iloc[ii, :].longitude, catalog_temp.iloc[ii, :].latitude, f'{event_list_to_compare[ii]}, M {catalog_temp.iloc[ii, :].magnitude}')
# %%
