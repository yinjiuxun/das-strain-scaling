#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import filter_event
from utility.regression import predict_magnitude, get_mean_magnitude
from utility.plotting import plot_magnitude_seaborn

#%%
# some parameters
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [2, 10]
weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise

random_test = True # whether to run the random test for transfered scaling

results_output_dir_list = []
regression_results_dir_list = []
peak_file_name_list = []
result_label_list = []
M_threshold_list = []
snr_threshold_list = []
vmax_list = []
region_text_list = []

#%% # Set result directory

#  ================== Arcata transfered random test ================== 
N_event_fit_list = range(2, 13)
N_test = 10

for N_event_fit in N_event_fit_list:
    for i_test in range(N_test):

        results_output_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate'
        peak_file_name = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
        result_label = 'transfer'
        regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
        snr_threshold = 5
        vmax = 50
        M_threshold = [2, 10]
        vmax = [2, 2] # for P and S
        region_text = 'Transfered scaling for Arcata'

        M_threshold_list.append(M_threshold)
        results_output_dir_list.append(results_output_dir)
        regression_results_dir_list.append(regression_results_dir)
        peak_file_name_list.append(peak_file_name)
        result_label_list.append(result_label)
        snr_threshold_list.append(snr_threshold)
        vmax_list.append(vmax)
        region_text_list.append(region_text) 

# #  ================== Sanriku transfered test ================== 
# N_event_fit_list = range(2, 10)
# N_test = 50
# for N_event_fit in N_event_fit_list:
#     for i_test in range(N_test):

#         results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
#         peak_file_name = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
#         result_label = 'transfer'
#         regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
#         snr_threshold = 5
#         vmax = 50
#         M_threshold = [0, 10]
#         vmax = [2, 2] # for P and S
#         region_text = 'Transfered scaling for Sanriku'

#         M_threshold_list.append(M_threshold)
#         results_output_dir_list.append(results_output_dir)
#         regression_results_dir_list.append(regression_results_dir)
#         peak_file_name_list.append(peak_file_name)
#         result_label_list.append(result_label)
#         snr_threshold_list.append(snr_threshold)
#         vmax_list.append(vmax)
#         region_text_list.append(region_text) 
        
# #  ================== LAX transfered test ================== 
# N_event_fit_list = range(2, 10)
# N_test = 50
# for N_event_fit in N_event_fit_list:
#     for i_test in range(20, N_test):

#         results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
#         peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
#         result_label = 'transfer'
#         regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
#         snr_threshold = 10
#         vmax = 50
#         M_threshold = [0, 10]
#         vmax = [2, 2] # for P and S
#         region_text = 'Transfered scaling for LAX'

#         M_threshold_list.append(M_threshold)
#         results_output_dir_list.append(results_output_dir)
#         regression_results_dir_list.append(regression_results_dir)
#         peak_file_name_list.append(peak_file_name)
#         result_label_list.append(result_label)
#         snr_threshold_list.append(snr_threshold)
#         vmax_list.append(vmax)
#         region_text_list.append(region_text)



#%%
for ii in range(len(peak_file_name_list)):
    
    peak_file_name = peak_file_name_list[ii]
    regression_results_dir = regression_results_dir_list[ii]
    result_label = result_label_list[ii]
    snr_threshold = snr_threshold_list[ii]
    M_threshold = M_threshold_list[ii]
    vmax = vmax_list[ii]
    region_text = region_text_list[ii]
    print(regression_results_dir)

    # # load results
    # peak_amplitude_df = pd.read_csv(peak_file_name)
    # peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
    # peak_amplitude_df = filter_event(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)
    # if 'Arcata' in results_output_dir:
    #     good_events_list = [73736021, 73739276, 73747016, 73751651, 73757961, 73758756, 
    #         73735891, 73741131, 73747621, 73747806, 73748011, 73753546, 73743421]
    #     peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.event_id.isin(good_events_list)]
    #     good_channel_list = np.concatenate([np.arange(0, 2750), np.arange(3450, 10000)])
    #     peak_amplitude_df = filter_event(peak_amplitude_df, channel_list=good_channel_list)

    site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{result_label}.csv')

    fig, ax = plt.subplots(1, 2, figsize=(16, 4), sharex=True, sharey=True)
    ax[0].plot(site_term_df.channel_id, site_term_df.site_term_P, '.')
    ax[0].set_title('P wave site calibration terms')
    ax[0].set_ylabel('site calibration \nterms (in log10)')
    ax[0].set_xlabel('channel #')
    ax[1].plot(site_term_df.channel_id, site_term_df.site_term_S, '.')
    ax[1].set_title('S wave site calibration terms')
    ax[1].set_xlabel('channel #')
    ax[0].grid()
    ax[1].grid()

    fig_dir = regression_results_dir + '/figures'
    mkdir(fig_dir)
    plt.savefig(fig_dir + f'/site_terms.png', bbox_inches='tight')
    plt.close('all')

# %%
temp = site_term_df.sort_values(['channel_id'])
# quickly show site terms, temporal use only
fig, ax = plt.subplots(2,1,figsize=(10, 15), sharex=True)
ax[0].plot(temp.channel_id, temp.site_term_P, '-')
ax[0].set_ylabel('P site term in log10')
ax[1].plot(temp.channel_id, temp.site_term_S, '-')
ax[1].set_ylabel('S site term in log10')
ax[1].set_xlabel('Channel number')

# %%
