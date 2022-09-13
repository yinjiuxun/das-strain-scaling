import numpy as np
import pandas as pd

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

# combine channel for regression
def combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number):
    if nearby_channel_number == -1: # when nearby_channel_number == -1, combined all channels!
        nearby_channel_number = DAS_index.max()+1
    temp1= np.arange(0, DAS_index.max()+1) # original channel number
    temp2 = temp1 // nearby_channel_number # combined channel number
    peak_amplitude_df['combined_channel_id'] = temp2[np.array(peak_amplitude_df.channel_id).astype('int')]
    return peak_amplitude_df

# load peak amplitude dataframe and add region label
def load_and_add_region(peak_file, region_label, snr_threshold, magnitude_threshold=None):
    peak_amplitude_df = pd.read_csv(peak_file)
    peak_amplitude_df['region'] = region_label # add the region label
    DAS_index = peak_amplitude_df.channel_id.unique().astype('int')
    #peak_amplitude_df = peak_amplitude_df.dropna()

    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.snrP >= snr_threshold) | (peak_amplitude_df.snrS >= snr_threshold)]

    if magnitude_threshold is not None:
        peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= magnitude_threshold[0]) & (peak_amplitude_df.magnitude <= magnitude_threshold[1])]
        
    return peak_amplitude_df,DAS_index

 # A function to add the event label to the peak ampliutde DataFrame
def add_event_label(peak_amplitude_df):
    '''A function to add the event label to the peak ampliutde DataFrame'''
    peak_amplitude_df['event_label'] = 0
    event_id_unique = peak_amplitude_df.event_id.unique()

    for i_event, event_id in enumerate(event_id_unique):
       peak_amplitude_df['event_label'][peak_amplitude_df['event_id'] == event_id] = i_event

    return peak_amplitude_df

# filter by channel number, only keep event with availble channels >= min_channel
def filter_by_channel_number(peak_amplitude_df, min_channel):
    """To remove the measurements from few channels (< min_channel)"""
    event_channel_count = peak_amplitude_df.groupby(['event_id'])['event_id'].count()
    channel_count = event_channel_count.values
    event_id = event_channel_count.index
    event_id = event_id[channel_count >= min_channel]

    return peak_amplitude_df[peak_amplitude_df['event_id'].isin(event_id)]


# split P and S data from regression separately.
def split_P_S_dataframe(peak_amplitude_df, snr_threshold=None):
    # use P and S separately to do the regression
    peak_amplitude_df_P = peak_amplitude_df[['event_id', 'peak_P', 'magnitude', 'distance_in_km', 'region_site', 'snrP']]
    peak_amplitude_df_S = peak_amplitude_df[['event_id', 'peak_S', 'magnitude', 'distance_in_km', 'region_site', 'snrS']]

    # Remove some extreme data outliers before fitting
    peak_amplitude_df_P = peak_amplitude_df_P.dropna()
    peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.peak_P>0]
    peak_amplitude_df_P = peak_amplitude_df_P.drop(peak_amplitude_df_P[(peak_amplitude_df_P.peak_P > 1e3)].index)

    peak_amplitude_df_S = peak_amplitude_df_S.dropna()
    peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.peak_S>0]
    peak_amplitude_df_S = peak_amplitude_df_S.drop(peak_amplitude_df_S[(peak_amplitude_df_S.peak_S > 1e3)].index)

    if snr_threshold:
        peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.snrP >=snr_threshold]
        peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.snrS >=snr_threshold]

    return peak_amplitude_df_P, peak_amplitude_df_S