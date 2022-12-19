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

random_test = False # whether to run the random test for transfered scaling

results_output_dir_list = []
regression_results_dir_list = []
peak_file_name_list = []
result_label_list = []
M_threshold_list = []
snr_threshold_list = []
vmax_list = []
region_text_list = []
catalog_file_list = []

#%% # Set result directory
if not random_test:
    # ================== multiple arrays ================== 
    results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
    peak_file_name = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/peak_amplitude_multiple_arrays.csv'
    catalog_file = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/catalog.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [120, 180] # for P and S
    region_text = 'California arrays'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)
    catalog_file_list.append(catalog_file)

    # single arrays
    #  ================== Ridgecrest ================== 
    results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [70, 100] # for P and S
    region_text = 'Ridgecrest'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== Long Valley N ================== 
    results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
    peak_file_name = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [35, 50] # for P and S
    region_text = 'Long Valley North'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== Long Valley S ================== 
    results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
    peak_file_name = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [20, 30] # for P and S
    region_text = 'Long Valley South'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== Sanriku fittd ================== 
    results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 5
    M_threshold = [2, 10]
    vmax = [2, 2] # for P and S
    region_text = 'Sanriku'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== LAX fittd ================== 
    results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [2, 2] # for P and S
    region_text = 'Los Angeles (LAX)'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== Arcata fittd ================== 
    results_output_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 20
    M_threshold = [2, 10]
    vmax = [2, 2] # for P and S
    region_text = 'Arcata'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

else:
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

    #  ================== Arcata transfered specified test ================== 
    results_output_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'transfer'
    regression_results_dir = results_output_dir + f'/transfer_regression_specified_smf{weight_text}_{min_channel}_channel_at_least/'
    snr_threshold = 10
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

    #  ================== Sanriku transfered test ================== 
    N_event_fit_list = range(2, 10)
    N_test = 50
    for N_event_fit in N_event_fit_list:
        for i_test in range(N_test):

            results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
            peak_file_name = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
            result_label = 'transfer'
            regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
            snr_threshold = 5
            vmax = 50
            M_threshold = [0, 10]
            vmax = [2, 2] # for P and S
            region_text = 'Transfered scaling for Sanriku'

            M_threshold_list.append(M_threshold)
            results_output_dir_list.append(results_output_dir)
            regression_results_dir_list.append(regression_results_dir)
            peak_file_name_list.append(peak_file_name)
            result_label_list.append(result_label)
            snr_threshold_list.append(snr_threshold)
            vmax_list.append(vmax)
            region_text_list.append(region_text) 
            
    #  ================== LAX transfered test ================== 
    N_event_fit_list = range(2, 10)
    N_test = 50
    for N_event_fit in N_event_fit_list:
        for i_test in range(20, N_test):

            results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
            peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
            result_label = 'transfer'
            regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
            snr_threshold = 10
            vmax = 50
            M_threshold = [0, 10]
            vmax = [2, 2] # for P and S
            region_text = 'Transfered scaling for LAX'

            M_threshold_list.append(M_threshold)
            results_output_dir_list.append(results_output_dir)
            regression_results_dir_list.append(regression_results_dir)
            peak_file_name_list.append(peak_file_name)
            result_label_list.append(result_label)
            snr_threshold_list.append(snr_threshold)
            vmax_list.append(vmax)
            region_text_list.append(region_text)



# %%
# calculate the DAS magnitude

for ii in range(len(peak_file_name_list)):
    
    peak_file_name = peak_file_name_list[ii]
    regression_results_dir = regression_results_dir_list[ii]
    result_label = result_label_list[ii]
    snr_threshold = snr_threshold_list[ii]
    M_threshold = M_threshold_list[ii]
    vmax = vmax_list[ii]
    region_text = region_text_list[ii]
    catalog_file = catalog_file_list[ii]
    print(regression_results_dir)

    # load catalog
    catalog = pd.read_csv(catalog_file)
    # load results
    peak_amplitude_df = pd.read_csv(peak_file_name)
    peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
    peak_amplitude_df = filter_event(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)
    if 'Arcata' in results_output_dir:
        good_events_list = [73736021, 73739276, 73747016, 73751651, 73757961, 73758756, 
            73735891, 73741131, 73747621, 73747806, 73748011, 73753546, 73743421]
        peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.event_id.isin(good_events_list)]
        good_channel_list = np.concatenate([np.arange(0, 2750), np.arange(3450, 10000)])
        peak_amplitude_df = filter_event(peak_amplitude_df, channel_list=good_channel_list)

        event_id_fit_P0 = [73736021, 73747016, 73735891, 73747621, 73747806, 73748011, 73739346, 73741131, 73743421] #[73736021, 73747016, 73747621] 
        peak_amplitude_df_P = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit_P0)]
        event_id_fit_S0 = [73736021, 73747016, 73735891, 73747621, 73747806, 73748011, 73739346, 73741131, 73743421] 
        peak_amplitude_df_S = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit_S0)]


    site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{result_label}.csv')

    try:
        regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{result_label}.pickle")
        # use the measured peak amplitude to estimate the magnitude
        if 'Arcata' in results_output_dir:
            peak_amplitude_df = peak_amplitude_df_P
        magnitude_P, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regP, site_term_df, wavetype='P')
        final_magnitude_P = get_mean_magnitude(peak_amplitude_df_temp, magnitude_P)
    except:
        print('No P regression results, skip...')
        regP, magnitude_P, peak_amplitude_df_temp, final_magnitude_P = None, None, None, None

    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{result_label}.pickle")
        if 'Arcata' in results_output_dir:
            peak_amplitude_df = peak_amplitude_df_S
        magnitude_S, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regS, site_term_df, wavetype='S')
        final_magnitude_S = get_mean_magnitude(peak_amplitude_df_temp, magnitude_S)
    except:
        print('No S regression results, skip...')
        regS, magnitude_S, peak_amplitude_df_temp, final_magnitude_S = None, None, None, None



# %%
# compare with different magnitude types
M_das = final_magnitude_S[['event_id', 'predicted_M']]
M_das = M_das.rename(columns={"predicted_M": "magnitude_das"})
M_compare = pd.merge(catalog, M_das, how='inner', on='event_id')

M_compare['magnitude_type']=M_compare['magnitude_type'].str.capitalize()
M_type_list = list(M_compare.magnitude_type.unique())
cmap = plt.cm.get_cmap('Set2', len(M_type_list))

fig, ax = plt.subplots(figsize=(12, 12))
for ii, type in enumerate(M_type_list):
    temp_df = M_compare[M_compare['magnitude_type']==type]
    ax.plot(temp_df['magnitude_das'], temp_df['magnitude'], '.', color=cmap(ii), label=type)
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.legend()
    

fig, ax = plt.subplots(figsize=(12, 3))
for ii, type in enumerate(M_type_list):
    temp_df = M_compare[M_compare['magnitude_type']==type]
    ax.plot(temp_df['magnitude_das'], temp_df['magnitude']-temp_df['magnitude_das'], '.', color=cmap(ii), markersize=10, label=type)
ax.hlines(xmin=0, xmax=8, y=[0], color='k')
ax.hlines(xmin=0, xmax=8, y=[0.5, -0.5], color='k', linewidth=1, linestyle='--')
ax.hlines(xmin=0, xmax=8, y=[1, -1], color='k', linewidth=0.5, linestyle='--')
ax.set_xlim(0, 8)
ax.set_ylim(-1.8, 1.8)
ax.legend()

# %%
