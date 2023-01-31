#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import shutil
import statsmodels.api as sm
import random
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import filter_by_channel_number, split_P_S_dataframe, filter_event
from utility.regression import fit_regression_transfer, get_std_of_site_terms

#%% 
# define some local function for convenience
def split_fit_and_predict(N_event_fit, peak_amplitude_df):
    """Randomly choose a few events to fit"""
    event_id_all =  peak_amplitude_df.event_id.unique()
    random.shuffle(event_id_all)
    event_id_fit = event_id_all[:N_event_fit] # event id list for regression fit
    event_id_predict = event_id_all[N_event_fit:] # event id list for regression prediction

    peak_amplitude_df_fit = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit)]
    peak_amplitude_df_predict = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_predict)]

    return list(event_id_fit), peak_amplitude_df_fit, list(event_id_predict), peak_amplitude_df_predict

def specify_fit_and_predict(event_id_fit, event_id_predict, peak_amplitude_df):
    """Specify the fit and predict events """
    peak_amplitude_df_fit = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit)]
    peak_amplitude_df_predict = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_predict)]

    return peak_amplitude_df_fit, peak_amplitude_df_predict    

def transfer_fitting(regP_pre, regS_pre, peak_amplitude_df_fit, weighted):
    """Transfer scaling to obtain site terms"""
    site_term_P = fit_regression_transfer(peak_amplitude_df_fit, regP_pre, wavetype='P', weighted=weighted, M_threshold=M_threshold, snr_threshold=snr_threshold_transfer, min_channel=min_channel)
    site_term_S = fit_regression_transfer(peak_amplitude_df_fit, regS_pre, wavetype='S', weighted=weighted, M_threshold=M_threshold, snr_threshold=snr_threshold_transfer, min_channel=min_channel)
    # get STD
    try:
        site_term_P = get_std_of_site_terms(peak_amplitude_df_fit, regP_pre, site_term_P, 'P')
    except:
        print('No P regression')
        pass

    try:
        site_term_S = get_std_of_site_terms(peak_amplitude_df_fit, regS_pre, site_term_S, 'S')
    except:
        print('No S regression')
        pass

    # combine P and S
    site_term_df = pd.merge(site_term_P, site_term_S, on=['channel_id', 'region'], how='outer')
    site_term_df['region'] = peak_amplitude_df_fit.region.unique()[0]
    site_term_df = site_term_df.iloc[:, [0, 3, 1, 2]]
    return site_term_df

#%%
# some parameters
min_channel = 100 # do regression only on events recorded by at least 100 channels

weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise
#%%
# Coefficients from previous results
previous_regression_dir = f'/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
regP_pre_path = previous_regression_dir + f"/P_regression_combined_site_terms_iter.pickle"
regS_pre_path = previous_regression_dir + f"/S_regression_combined_site_terms_iter.pickle"

regP_pre = sm.load(regP_pre_path)
regS_pre = sm.load(regS_pre_path)

#%%
results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
snr_threshold_transfer = 5
M_threshold = [2, 10]

peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
peak_amplitude_df = filter_event(peak_amplitude_df, snr_threshold=snr_threshold_transfer, min_channel=min_channel, M_threshold=M_threshold)
event_id_all =  peak_amplitude_df.event_id.unique()

random.seed(212)
N_event_fit_list = [10]#range(2, 13)
N_test = 1

for N_event_fit in N_event_fit_list:
    for i_test in range(N_test):
        print(f'================= {N_event_fit} to fit, {i_test}th test =====================')

        # Randomly choose a few events to fit
        event_id_fit, peak_amplitude_df_fit, event_id_predict, peak_amplitude_df_predict = split_fit_and_predict(N_event_fit, peak_amplitude_df)
        event_id_fit_P0, event_id_fit_S0 = event_id_fit, event_id_fit
                
        # Transfer scaling to obtain site terms
        site_term_df = transfer_fitting(regP_pre, regS_pre, peak_amplitude_df_fit, weighted)

        # make output directory and output results
        results_output_dir = results_output_dir
        regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least'
        mkdir(regression_results_dir)
        
        regression_results_dir = regression_results_dir + f'/{N_event_fit}_fit_events_{i_test}th_test'
        mkdir(regression_results_dir)

        site_term_df.to_csv(regression_results_dir + '/site_terms_transfer.csv', index=False)

        # output the event id list of fit and predict events
        np.savez(regression_results_dir + '/transfer_event_list.npz', 
            event_id_fit_P=event_id_fit_P0, event_id_fit_S=event_id_fit_S0, event_id_predict=event_id_predict)

        # also copy the regression results to the results directory
        shutil.copyfile(regP_pre_path, regression_results_dir + '/P_regression_combined_site_terms_transfer.pickle')
        shutil.copyfile(regS_pre_path, regression_results_dir + '/S_regression_combined_site_terms_transfer.pickle')




