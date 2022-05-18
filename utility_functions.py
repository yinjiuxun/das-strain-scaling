import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import h5py
import dateutil
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


# Functions to save and load event h5 files
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
   """
   """
   with h5py.File(fn, 'r') as fid:
       data = fid['data'][:]
       info = {}
       for key in fid['data'].attrs.keys():
           info[key] = fid['data'].attrs[key]
       info2 = {}
       if 'begin_time' in info.keys():
           info2['begTime'] = dateutil.parser.parse(info['begin_time'])
       if 'end_time' in info.keys():
           info2['endTime'] = dateutil.parser.parse(info['end_time'])
       if 'event_time' in info.keys():
           info2['time'] = dateutil.parser.parse(info['event_time'])
       info2['nt'] = data.shape[0]
       info2['nx'] = data.shape[1]
       info2['dx'] = info['dx_m']
       info2['dt'] = info['dt_s']
   return data, info2

def load_event_data(data_path, eq_id):
    with h5py.File(data_path + '/' + str(eq_id) + '.h5', 'r') as fid:
       data = fid['data'][:]
       info = {}
       for key in fid['data'].attrs.keys():
           info[key] = fid['data'].attrs[key]
       info2 = {}
       if 'begin_time' in info.keys():
           info2['begTime'] = dateutil.parser.parse(info['begin_time'])
       if 'end_time' in info.keys():
           info2['endTime'] = dateutil.parser.parse(info['end_time'])
       if 'event_time' in info.keys():
           info2['event_time'] = dateutil.parser.parse(info['event_time'])
       info2['nt'] = data.shape[0]
       info2['nx'] = data.shape[1]
       info2['dx'] = info['dx_m']
       info2['dt'] = info['dt_s']
       return data, info2

def load_phase_pick(pick_path, eq_id, das_time, channel, time_range=None, include_nan=False):
    picks = pd.read_csv(pick_path + f'/{eq_id}.csv')

    picks_P = picks[picks.phase_type == 'P']
    picks_S = picks[picks.phase_type == 'S']
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
        picks_P_time[picks_P.channel_index] = das_time[picks_P.phase_index]
        picks_S_time[picks_S.channel_index] = das_time[picks_S.phase_index]
        channel_P, channel_S = channel, channel

    else:
        picks_P_time = das_time[picks_P.phase_index]
        channel_P = channel[picks_P.channel_index]

        picks_S_time = das_time[picks_S.phase_index]
        channel_S = channel[picks_S.channel_index]

    return picks_P_time, channel_P, picks_S_time, channel_S
   

def combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number):
    if nearby_channel_number == -1: # when nearby_channel_number == -1, combined all channels!
        nearby_channel_number = DAS_index.max()+1
    temp1= np.arange(0, DAS_index.max()+1) # original channel number
    temp2 = temp1 // nearby_channel_number # combined channel number
    peak_amplitude_df['combined_channel_id'] = temp2[np.array(peak_amplitude_df.channel_id).astype('int')]
    return peak_amplitude_df

def load_and_add_region(peak_file, region_label, snr_threshold):
    peak_amplitude_df = pd.read_csv(peak_file)
    peak_amplitude_df['region'] = region_label # add the region label
    DAS_index = peak_amplitude_df.channel_id.unique().astype('int')
    peak_amplitude_df = peak_amplitude_df.dropna()

    # Mammoth data contains the snrP and snrS, so if these two columns exist, only keep data with higher SNR
    if 'snrP' in peak_amplitude_df.columns:
        peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrP >= snr_threshold]

    if 'snrS' in peak_amplitude_df.columns:
        peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrS >= snr_threshold]
        
    return peak_amplitude_df,DAS_index

 
def add_event_label(peak_amplitude_df):
    '''A function to add the event label to the peak ampliutde DataFrame'''
    peak_amplitude_df['event_label'] = 0
    event_id_unique = peak_amplitude_df.event_id.unique()

    for i_event, event_id in enumerate(event_id_unique):
       peak_amplitude_df['event_label'][peak_amplitude_df['event_id'] == event_id] = i_event

    return peak_amplitude_df

#%% Functions used to for the regression
def model_parameters_df(reg, combined_channel_number, digits=3):
    magnitude = round(reg.params[-2],digits)
    distance = round(reg.params[-1],digits)
    magnitude_err = round(np.sqrt(reg.cov_params().iloc[-2][-2]),digits)
    distance_err = round(np.sqrt(reg.cov_params().iloc[-1][-1]),digits)
    parameter_df = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
    data = [[combined_channel_number, magnitude, distance, magnitude_err, distance_err]])
    return parameter_df

def fit_regression_magnitude_range(combined_channel_number_list, M_threshold, results_output_dir, regression_results_dir, regression_text):
    # DataFrame to store parameters for all models
    P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err']) 
    S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'])

    for nearby_channel_number in combined_channel_number_list:
        if nearby_channel_number == -1:
            peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_region_site_all.csv')
            peak_amplitude_df_M = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
            regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df_M).fit()
            regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df_M).fit()
        else:
            peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
            peak_amplitude_df_M = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
        # %% Now can fit the data with different regional site terms
            regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_M).fit()
            regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_M).fit()

        regP.save(regression_results_dir + f"/P_regression_region_site_terms_{nearby_channel_number}chan.pickle")
        regS.save(regression_results_dir + f"/S_regression_region_site_terms_{nearby_channel_number}chan.pickle")

    # output to text files
        with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
            text_file.write(regP.summary().as_text())
        with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
            text_file.write(regS.summary().as_text())

    # Store the parameters 
        P_parameters_comparison = pd.concat([P_parameters_comparison, model_parameters_df(regP, nearby_channel_number)], axis=0)
        S_parameters_comparison = pd.concat([S_parameters_comparison, model_parameters_df(regS, nearby_channel_number)], axis=0)

    P_parameters_comparison.to_csv(regression_text + '/parameter_comparison_P.txt', index=False, sep='\t')
    S_parameters_comparison.to_csv(regression_text + '/parameter_comparison_S.txt', index=False, sep='\t')
