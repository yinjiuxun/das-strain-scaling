#%%
# Import modules
import pandas as pd
#from sep_util import read_file
import numpy as np
import tqdm
import glob
import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

import warnings
from obspy.geodetics import locations2degrees

# Plotting
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
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

def extract_peak_amplitude(event_folder, peak_amplitude_dir, ii_region, data_path, pick_path, catalog, das_info, P_window_list, S_window_list, snr_window, i_eq):
    matplotlib.rcParams.update(params) # Set up the plotting parameters
    warnings.filterwarnings('ignore')

    try:
        event_now = catalog.iloc[i_eq, :]
        if 'eventID' in event_now.keys():
            event_now = event_now.rename({"eventID": "event_id"})

        event_data, event_info = load_event_data(data_path, event_now.event_id)

    except:
        print(f'event data {event_now.event_id} was not found, skip...')
        return

    if event_data.shape[1] > das_info.shape[0]:
        event_data = event_data[:, das_info['index']]

    nt = event_data.shape[0]
    das_time = np.arange(0, nt) * event_info['dt_s']

    tt_1d_path = pick_path
    tt_1d = pd.read_csv(tt_1d_path + f'/1D_tt_{event_now.event_id}.csv')
    pick_P = np.array(tt_1d.P_arrival)
    channel_P = np.array(tt_1d['index'])
    pick_S = np.array(tt_1d.S_arrival)
    channel_S = np.array(tt_1d['index'])

    
    # Calculate the SNR based on the P arrival time
    event_arrival_P_temp = pick_P[np.newaxis, :]
    event_arrival_S_temp = pick_S[np.newaxis, :]
    twin_noise_P = [event_arrival_P_temp-1-snr_window, event_arrival_P_temp-1]
    twin_noise_S = [event_arrival_S_temp-1-snr_window, event_arrival_S_temp-1]
    twin_signal_P = [event_arrival_P_temp, event_arrival_P_temp + snr_window]
    twin_signal_S = [event_arrival_S_temp, event_arrival_S_temp + snr_window]
    snrP = calculate_SNR(das_time, event_data, twin_noise_P, twin_signal_P)
    snrS = calculate_SNR(das_time, event_data, twin_noise_S, twin_signal_S)


    # Extract the maximum given the P and S arrival time
    max_P_amplitude_list = []
    max_P_time_list = []
    max_S_amplitude_list = []
    max_S_time_list = []

    for P_window in P_window_list:
        P_max_window = np.nanmin(np.array([pick_P+P_window-0.5, pick_S-0.5]), axis=0)
        max_P_amplitude, max_P_time, _ = extract_maximum_amplitude(das_time, event_data, pick_P, P_max_window)
        max_P_amplitude_list.append(max_P_amplitude)
        max_P_time_list.append(max_P_time)
        if P_window == P_window_list[0]:
            peak_P_time = max_P_time.copy()

    for S_window in S_window_list:
        max_S_amplitude, max_S_time, _ = extract_maximum_amplitude(das_time, event_data, pick_S, pick_S+S_window)
        max_S_amplitude_list.append(max_S_amplitude)
        max_S_time_list.append(max_S_time)
        if S_window == S_window_list[0]:
            peak_S_time = max_S_time.copy()

    # Convert to array
    max_P_amplitude_list = np.array(max_P_amplitude_list).T
    max_P_time_list = np.array(max_P_time_list).T
    max_S_amplitude_list = np.array(max_S_amplitude_list).T
    max_S_time_list = np.array(max_S_time_list).T

    # Source information from catalog
    event_id_list = (np.ones(das_info.shape[0]) * event_now.event_id).astype('int')
    distance_list = locations2degrees(das_info.latitude, das_info.longitude, event_now.latitude, event_now.longitude) * 113
    magnitude_list = np.ones(das_info.shape[0]) * event_now.magnitude
    channel_id_list = np.array(das_info['index'])
    # combined into one 
    all_combined1 = np.array([event_id_list, magnitude_list, channel_id_list, distance_list, snrP, snrS]).T

    all_combined = np.concatenate([all_combined1, 
                                max_P_amplitude_list, max_P_time_list, max_S_amplitude_list, max_S_time_list], axis=1)

    # Convert to DataFrame
    peak_amplitude_df = pd.DataFrame(data=all_combined, 
                                    columns=['event_id', 'magnitude', 'channel_id', 'distance_in_km', 'snrP', 'snrS', 
                                    'peak_P', 'peak_P_1s', 'peak_P_3s', 'peak_P_4s', 
                                    'peak_P_time', 'peak_P_time_1s', 'peak_P_time_3s', 'peak_P_time_4s',
                                    'peak_S', 'peak_S_4s', 'peak_S_6s', 'peak_S_8s', 'peak_S_10s', 
                                    'peak_S_time', 'peak_S_time_4s', 'peak_S_time_6s', 'peak_S_time_8s', 'peak_S_time_10s'])

    # # Remove NaNs
    # peak_amplitude_df = peak_amplitude_df.dropna()
    # Adjust the format
    peak_amplitude_df.event_id = peak_amplitude_df.event_id.astype('int')
    # Store to csv
    peak_amplitude_df.to_csv(peak_amplitude_dir + f'/{event_now.event_id}.csv',index=False)

    #%
    # Show data
    fig, ax = plt.subplots(figsize=(10,5))
    # Main
    gca = ax
    show_event_data(event_data, das_time, gca, pclip=95)
    gca.plot(channel_P, pick_P, '.', markersize=3, color='green', label='P')
    gca.scatter(channel_S, pick_S + peak_S_time, s=2, c=snrS, cmap='jet')
    cbar = gca.scatter(channel_P, pick_P + peak_P_time, s=2, c=snrP, cmap='jet')
    plt.colorbar(cbar, label='SNR')
    gca.plot(channel_S, pick_S, '.', markersize=3, color='cyan', label='S')
    gca.legend()
    # gca.plot(tt_1d.index, tt_1d.P_arrival, '--g')
    # gca.plot(tt_1d.index, tt_1d.S_arrival, '-g')
    # gca.set_ylim(30, 120)

    gca.set_title(f'ID {event_now.event_id}, M {event_now.magnitude}')
    plt.savefig(peak_amplitude_dir + f'/figures/{event_now.event_id}.png', bbox_inches='tight')
    plt.close('all')


# %%
# Setup the paths
event_folder_list = ['/kuafu/EventData/Curie', '/kuafu/EventData/Arcata_Spring2022']
peak_amplitude_dir_list = ['/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events', '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']

# Extract the peak amplitude given the picking 
for ii_region in [0]:#[0, 1, 2, 3]:
    event_folder, peak_amplitude_dir = event_folder_list[ii_region], peak_amplitude_dir_list[ii_region]
    mkdir(peak_amplitude_dir)
    mkdir(peak_amplitude_dir + '/figures')
    print('='*10 + peak_amplitude_dir + '='*10)

    # Event waveform directory, phase picking directory
    data_path = event_folder + '/data'
    pick_path = event_folder + '/theoretical_arrival_time_calibrated' # directory of ML phase picker

    # Load the catalog, filter events, load waveforms
    catalog = pd.read_csv(event_folder + '/catalog.csv')
    das_info = pd.read_csv(event_folder + '/das_info.csv')

    # 
    # parameters of time window
    P_window_list = [2, 1, 3, 4] # 100 here mean S_pick-0.5
    S_window_list = [2, 4, 6, 8, 10]
    snr_window = 2

    n_eq = catalog.shape[0]

    # Non-parallel version for testing purpose
    # for i_eq in range(100):
    # i_eq = 7110
    # extract_peak_amplitude(event_folder, peak_amplitude_dir, ii_region, data_path, pick_path, catalog, das_info, P_window_list, S_window_list, snr_window, i_eq)

    # Parallel extracting peak amplitude
    with tqdm_joblib(tqdm(desc="File desampling", total=n_eq)) as progress_bar:
        Parallel(n_jobs=Ncores)(delayed(extract_peak_amplitude)(event_folder, peak_amplitude_dir, ii_region, data_path, pick_path, catalog, das_info, P_window_list, S_window_list, snr_window, i_eq) for i_eq in range(n_eq))

# %%
# Combine all the individual peak amplitude files into one for regression
event_folder_list = ['/kuafu/EventData/Curie', '/kuafu/EventData/Arcata_Spring2022']
peak_amplitude_dir_list = ['/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events', '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']


for ii_region in [0]:
    peak_amplitude_dir = peak_amplitude_dir_list[ii_region]

    print('='*10 + peak_amplitude_dir + '='*10)
    temp = glob.glob(peak_amplitude_dir + '/*.csv')
    temp_df = pd.concat(map(pd.read_csv, temp), ignore_index=True)

    # calibrate SNR
    ii = ~temp_df.snrP.isna()
    temp_df.snrS[ii] = temp_df.snrP[ii] + temp_df.snrS[ii]

    temp_df.to_csv(peak_amplitude_dir + '/peak_amplitude.csv', index=False)
# %%
# Calibrate distance to hypocentral distance
das_pick_file_folder_list = ['/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events', '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']

catalog_file_list = ['/kuafu/EventData/Curie/catalog.csv','/kuafu/EventData/Arcata_Spring2022/catalog.csv']

region_list = ['curie', 'arcata']

das_pick_file_name = 'peak_amplitude.csv'

for ii in [0]:#range(4):
    das_pick_file_folder = das_pick_file_folder_list[ii]
    catalog_file = catalog_file_list[ii]
    print(f'=========== Working on {das_pick_file_folder} ===============')

    peak_amplitude_df = pd.read_csv(das_pick_file_folder + '/' + das_pick_file_name)
    catalog = pd.read_csv(catalog_file)

    # combine the two dataframe to include the event depth information
    catalog_depth = catalog.loc[:, ['event_id', 'depth_km']]
    peak_amplitude_df = pd.merge(peak_amplitude_df, catalog_depth, on="event_id")
    peak_amplitude_df['calibrated_distance_in_km'] = np.sqrt(peak_amplitude_df.depth_km**2 + peak_amplitude_df.distance_in_km**2)
    

    if ii==3:
        temp1 = np.array([0, 1, 25, 2, 3, 26])
        temp2 = np.arange(4, 25)
    else:
        temp1 = np.array([0, 1, 24, 2, 3, 25])
        temp2 = np.arange(4, 24)
    
    reorder_index = np.concatenate((temp1, temp2), axis=0)
    peak_amplitude_df = peak_amplitude_df.iloc[:, reorder_index]
    peak_amplitude_df['region'] = region_list[ii]

    peak_amplitude_df.to_csv(das_pick_file_folder + '/calibrated_' + das_pick_file_name, index=False)
# %%
