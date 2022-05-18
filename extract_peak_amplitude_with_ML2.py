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

import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

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


# calculate the SNR given the P arrival time
def calculate_SNR(time_list, data_matrix, twin_noise, twin_signal):
    '''calculate the SNR given the noise and signal time window list [begin, end] for each channel'''
    time_list = time_list[:, np.newaxis]

    noise_index = (time_list < twin_noise[1]) & (time_list >= twin_noise[0]) # noise index
    signal_index = (time_list <= twin_signal[1]) & (time_list >= twin_signal[0]) # signal index

    noise_matrix = data_matrix.copy()
    signal_matrix = data_matrix.copy()
    noise_matrix[~noise_index] = np.nan
    signal_matrix[~signal_index] = np.nan

    noise_power = np.nanmean(noise_matrix ** 2, axis=0)
    signal_power = np.nanmean(signal_matrix ** 2, axis=0)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def load_phase_pick(pick_path, eq_id, das_time, channel, time_range=None, include_nan=False):
    picks = pd.read_csv(pick_path + f'/{eq_id}.csv')

    picks_P = picks[picks.phase_type == 'P'].drop_duplicates(subset=['channel_index'], keep='first')
    picks_S = picks[picks.phase_type == 'S'].drop_duplicates(subset=['channel_index'], keep='first')


    # Adding restriction on the time
    if time_range is not None:
        dt = das_time[1] - das_time[0]
        picks_P = picks_P[(picks_P.phase_type == 'P') & 
                            (picks_P.phase_index <= time_range[1]/dt) & 
                            (picks_P.phase_index >= time_range[0]/dt)]

        picks_S = picks_S[(picks_S.phase_type == 'S') & 
                            (picks_S.phase_index <= time_range[3]/dt) & 
                            (picks_S.phase_index >= time_range[2]/dt)]

    if include_nan:
        picks_P_time = np.ones(channel.shape) * np.nan
        picks_S_time = np.ones(channel.shape) * np.nan
        ii_p = channel.isin(picks_P.channel_index.unique())#picks_P.channel_index.isin(channel)
        ii_s = channel.isin(picks_S.channel_index.unique())#picks_S.channel_index.isin(channel)

        ii_p_picks = picks_P.channel_index.isin(channel)
        ii_s_picks = picks_S.channel_index.isin(channel)

        picks_P_time[ii_p] = das_time[picks_P.phase_index[ii_p_picks]]
        picks_S_time[ii_s] = das_time[picks_S.phase_index[ii_s_picks]]
        channel_P, channel_S = channel, channel

    else:
        picks_P_time = das_time[picks_P.phase_index]
        channel_P = channel[picks_P.channel_index]

        picks_S_time = das_time[picks_S.phase_index]
        channel_S = channel[picks_S.channel_index]

    return picks_P_time, channel_P, picks_S_time, channel_S


Ridgecrest_conversion_factor = 1550.12 / (0.78 * 4 * np.pi * 1.46 * 8)
# %%
# Setup the paths
event_folder = '/kuafu/EventData/Ridgecrest'
# event_folder = '/kuafu/EventData/Mammoth_north'
# event_folder = '/kuafu/EventData/Mammoth_south'

# Output directory of peak amplitude for each event
peak_amplitude_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events'
# peak_amplitude_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events'
# peak_amplitude_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events'


event_folder_list = ['/kuafu/EventData/Ridgecrest', '/kuafu/EventData/Mammoth_north', '/kuafu/EventData/Mammoth_south']
peak_amplitude_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events']

for ii_region in [1]:
    event_folder, peak_amplitude_dir = event_folder_list[ii_region], peak_amplitude_dir_list[ii_region]

    print('='*10 + peak_amplitude_dir + '='*10)

    # Event waveform directory, phase picking directory
    data_path = event_folder + '/data'
    pick_path = event_folder + '/picks_phasenet_das' # directory of ML phase picker

    # Load the catalog, filter events, load waveforms
    catalog = pd.read_csv(event_folder + '/catalog.csv')
    das_info = pd.read_csv(event_folder + '/das_info.csv')

    # 
    # parameters of time window
    P_window_list = [2, 1, 3, 4] # 100 here mean S_pick-0.5
    S_window_list = [2, 4, 6, 8, 10]
    snr_window = 2

    # load event waveform
    for i_eq in tqdm.tqdm(range(1514,catalog.shape[0])): # 0 - 1000 are still left.
    #i_eq = 1199
        if catalog.iloc[i_eq, :].magnitude >= 3:
            continue

        # try:
        event_now = catalog.iloc[i_eq, :]
        if 'eventID' in event_now.keys():
            event_now = event_now.rename({"eventID": "event_id"})

        event_data, event_info = load_event_data(data_path, event_now.event_id)

        xxx
        if event_data.shape[1] > das_info.shape[0]:
            event_data = event_data[:, das_info.index]

        das_time = np.arange(0, event_info['nt']) * event_info['dt']


        if ii_region == 0:
            tt_1d_path = event_folder + '/theoretical_arrival_time'
            tt_1d = pd.read_csv(tt_1d_path + f'/1D_tt_{event_now.event_id}.csv')


        elif ii_region == 1:
            tt_1d_path = '/kuafu/jxli/TemplateMatch/Events/Mammoth/North/TravelTime'
            tt_1d = pd.read_csv(tt_1d_path + f'/{event_now.event_id}.table')
            tt_1d.rename(columns = {'ichan':'index', 'tp':'P_arrival', 'ts':'S_arrival'}, inplace = True)
            tt_1d.P_arrival = tt_1d.P_arrival + 30
            tt_1d.S_arrival = tt_1d.S_arrival + 30

        elif ii_region == 2:
            tt_1d_path = '/kuafu/jxli/TemplateMatch/Events/Mammoth/South/TravelTime'
            tt_1d = pd.read_csv(tt_1d_path + f'/{event_now.event_id}.table')
            tt_1d.rename(columns = {'ichan':'index', 'tp':'P_arrival', 'ts':'S_arrival'}, inplace = True)
            tt_1d.P_arrival = tt_1d.P_arrival + 30
            tt_1d.S_arrival = tt_1d.S_arrival + 30

        min_P_1d = tt_1d.P_arrival.min()
        min_S_1d = tt_1d.S_arrival.min()   

        time_range = (min_P_1d-15, min_P_1d+15, min_S_1d-15, min_S_1d+15)#(25, 40, 25, 60)
        pick_P, channel_P, pick_S, channel_S = load_phase_pick(pick_path, event_now.event_id, das_time, das_info.index, time_range, include_nan=True)
        # except:
        #     continue
        
        # Calculate the SNR based on the P arrival time
        event_arrival_P_temp = pick_P[np.newaxis, :]
        event_arrival_S_temp = pick_S[np.newaxis, :]
        twin_noise = [event_arrival_P_temp-1-snr_window, event_arrival_P_temp-1]
        twin_signal_P = [event_arrival_P_temp, event_arrival_P_temp + snr_window]
        twin_signal_S = [event_arrival_S_temp, event_arrival_S_temp + snr_window]
        snrP = calculate_SNR(das_time, event_data, twin_noise, twin_signal_P)
        snrS = calculate_SNR(das_time, event_data, twin_noise, twin_signal_S)


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
        channel_id_list = np.array(das_info.index)
        # combined into one 
        all_combined1 = np.array([event_id_list, magnitude_list, channel_id_list, distance_list, snrP, snrS]).T

        all_combined = np.concatenate([all_combined1, 
                                    max_P_amplitude_list, max_S_amplitude_list, max_P_time_list, max_S_time_list], axis=1)

        # Convert to DataFrame
        peak_amplitude_df = pd.DataFrame(data=all_combined, 
                                        columns=['event_id', 'magnitude', 'channel_id', 'distance_in_km', 'snrP', 'snrS', 
                                        'peak_P', 'peak_P_1s', 'peak_P_3s', 'peak_P_4s', 
                                        'peak_P_time', 'peak_P_time_1s', 'peak_P_time_3s', 'peak_P_time_4s',
                                        'peak_S', 'peak_S_4s', 'peak_S_6s', 'peak_S_8s', 'peak_S_10s', 
                                        'peak_S_time', 'peak_S_time_4s', 'peak_S_time_6s', 'peak_S_time_8s', 'peak_S_time_10s'])

        # Remove NaNs
        peak_amplitude_df = peak_amplitude_df.dropna()
        # Adjust the format
        peak_amplitude_df.event_id = peak_amplitude_df.event_id.astype('int')
        # Store to csv
        peak_amplitude_df.to_csv(peak_amplitude_dir + f'/{event_now.event_id}.csv',index=False)


        #%%
        # Show data
        fig, ax = plt.subplots(figsize=(12,6))
        # Main
        gca = ax
        show_event_data(event_data, das_time, gca, pclip=95)
        gca.plot(channel_P, pick_P, '.k', markersize=2)
        gca.scatter(channel_S, pick_S + peak_S_time, s=10, c=snrS, cmap='jet')
        cbar = gca.scatter(channel_P, pick_P + peak_P_time, s=10, c=snrP, cmap='jet')
        plt.colorbar(cbar, label='SNR')
        gca.plot(channel_S, pick_S, '.k', markersize=2)

        gca.plot(tt_1d.index, tt_1d.P_arrival, '--g')
        gca.plot(tt_1d.index, tt_1d.S_arrival, '-g')
        gca.set_ylim(time_range[0], time_range[-1]+10)
        gca.set_title(f'ID {event_now.event_id}, M {event_now.magnitude}')
        plt.savefig(peak_amplitude_dir + f'/figures/{event_now.event_id}.png', bbox_inches='tight')
        plt.close('all')
        # except:
        #     continue



# %%
# # Combine individual event csv files to form a large one
# import glob

# results_folder = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling'
# peak_amplitude_dir = results_folder + '/peak_amplitude_events'

# # results_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
# # peak_amplitude_dir = results_folder + '/peak_amplitude_events'

# #%%
# temp = glob.glob(peak_amplitude_dir + '/*.csv')
# # %%
# temp_df = pd.concat(map(pd.read_csv, temp), ignore_index=True)
# temp_df.to_csv(results_folder + '/Ridgecrest_peak_amplitude.csv', index=False)

# #%%
# #temp_df = pd.read_csv(results_folder + '/Ridgecrest_peak_amplitude.csv')
# temp_df = pd.read_csv(results_folder + '/Ridgecrest_peak_amplitude.csv')
# import seaborn as sns
# peak_amplitude_df = temp_df[(temp_df.snrP > 0) & (temp_df.snrS > 0)]#& (temp_df.magnitude >=3)]

# # plot data statistics
# peak_amplitude_df_temp = peak_amplitude_df.iloc[::1, :]
# peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
# peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P_1s.astype('float')*1e3)
# peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float')*1e3)
# #peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S

# plt.figure(figsize=(14,8))
# g = sns.pairplot(peak_amplitude_df_temp[['magnitude','log10(distance)', 'log10(peak_P)','log10(peak_S)']], kind='hist', diag_kind="kde", corner=True)
# g.axes[2,0].set_ylim((2,5))
# g.axes[2,1].set_ylim((2,5))
# g.axes[3,0].set_ylim((2,5))
# g.axes[3,1].set_ylim((2,5))
# g.axes[3,2].set_ylim((2,5))
# g.axes[3,2].set_xlim((2,5))

# # %%
# peak_amplitude_df_RC = peak_amplitude_df_temp.copy()
# # %%
# peak_amplitude_df_LV_N = peak_amplitude_df_temp.copy()
# # %%
# fig, gca = plt.subplots(figsize=(8, 6))
# gca.plot(peak_amplitude_df_RC['log10(distance)']+ np.log10(113), peak_amplitude_df_RC['log10(peak_S)'] - np.log10(Ridgecrest_conversion_factor), 'ro', label='Ridgecrest')
# gca.plot(peak_amplitude_df_LV_N['log10(distance)']+ np.log10(113), peak_amplitude_df_LV_N['log10(peak_S)'], 'bo', label='Mammoth N')
# gca.set_ylim(0.5, 3.5)
# gca.set_xlabel('log10(Distance)')
# gca.set_ylabel('Peak S strain rate')
# gca.legend()
# %%
