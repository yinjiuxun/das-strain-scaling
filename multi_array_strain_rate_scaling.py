#%% import modules
import pandas as pd
#from sep_util import read_file
import numpy as np
import matplotlib.pyplot as plt

# import utility functions
from utility.general import mkdir
from utility.processing import combined_regions_for_regression, filter_event
from utility.regression import store_regression_results, fit_regression_iteration

#%% 
# some parameters
snr_threshold = 10
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [0, 10]
combined_channel_number_list = [-1]#[-1, 100, 50, 20, 10]#[10, 20, 50, 100, -1] # -1 means the constant model

# result directory
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
peak_file_name = 'peak_amplitude_multiple_arrays.csv'
mkdir(results_output_dir)

#%% 
# Preprocess the data file: combining different channels etc.
preprocess_needed = False # if true, combined different regions data to produce the combined data file
peak_file_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events/calibrated_peak_amplitude.csv',
                '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events/calibrated_peak_amplitude.csv',
                '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events/calibrated_peak_amplitude.csv']
# region_list = ['ridgecrest', 'mammothS', 'mammothN']

if preprocess_needed:
    peak_amplitude_df = combined_regions_for_regression(peak_file_list)
    # combined_regions_for_regression(peak_file_list, region_list, combined_channel_number_list, results_output_dir,
    #                                 snr_threshold, M_threshold, apply_calibrated_distance=True) # TODO: may not need the combined channel list
    peak_amplitude_df=filter_event(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)
    peak_amplitude_df.to_csv(results_output_dir + f'/{peak_file_name}', index=False)

#%% Iteratively fitting
regression_results_dir = results_output_dir + f'/iter_regression_results_smf_weighted_{min_channel}_channel_at_least'
mkdir(regression_results_dir)

peak_amplitude_df = pd.read_csv(results_output_dir + f'/{peak_file_name}')

peak_amplitude_df = peak_amplitude_df[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km', 
                                    'snrP', 'snrS', 'peak_P', 'peak_S', 'region']] #, 'combined_channel_id', 'event_label', 'region_site'

peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.peak_P<=0].index)
peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.peak_S<=0].index)

weighted = 'wls'
n_iter = 50
rms_epsilon = -0.1 # percentage of rms improvement, if smaller, stop iteration
show_figure = True
regP, regS, site_term_df = fit_regression_iteration(peak_amplitude_df, weighted=weighted, 
                            M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel, 
                            n_iter=n_iter, rms_epsilon=rms_epsilon, show_figure=show_figure)

# store the regression results
results_file_name = "regression_combined_site_terms_iter"
store_regression_results(regP, regression_results_dir, results_filename=f"/P_{results_file_name}")
store_regression_results(regS, regression_results_dir, results_filename=f"/S_{results_file_name}")
site_term_df.to_csv(regression_results_dir + f'/site_terms_iter.csv', index=False)


#%% 
# Apply to single array
results_output_dirs = ["/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate",
                       "/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South",
                       "/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North", 
                       '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate']
                       #TODO: Sanriku needs more work!

for i_region in [2]:#range(len(results_output_dirs)):

    results_output_dir = results_output_dirs[i_region]
    regression_results_dir = results_output_dir + f'/iter_regression_results_smf_weighted_{min_channel}_channel_at_least'
    mkdir(regression_results_dir)

    peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')

    peak_amplitude_df = peak_amplitude_df[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km', 
                                        'snrP', 'snrS', 'peak_P', 'peak_S', 'region']] 

    peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.peak_P<=0].index)
    peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.peak_S<=0].index)

    weighted = 'wls'
    n_iter = 50
    rms_epsilon = -0.2 # percentage of rms improvement, if smaller, stop iteration
    show_figure = True
    regP, regS, site_term_df = fit_regression_iteration(peak_amplitude_df, weighted=weighted, 
                                M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel, 
                                n_iter=n_iter, rms_epsilon=rms_epsilon, show_figure=show_figure)

    # store the regression results
    results_file_name = "regression_combined_site_terms_iter"
    store_regression_results(regP, regression_results_dir, results_filename=f"/P_{results_file_name}")
    store_regression_results(regS, regression_results_dir, results_filename=f"/S_{results_file_name}")
    site_term_df.to_csv(regression_results_dir + f'/site_terms_iter.csv', index=False)


#%% 
# compare predicted and measured
peak_amplitude_df_temp = filter_event(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)
peak_amplitude_df_temp = pd.merge(peak_amplitude_df, site_term_df, how='outer', left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])
peak_P_predicted = 10**(regP.predict(peak_amplitude_df_temp) + peak_amplitude_df_temp.site_term_P)

plt.loglog(peak_amplitude_df_temp.peak_P, peak_P_predicted, '.', alpha=0.005)
plt.loglog([1e-3, 1e2], [1e-3, 1e2], '-k')
plt.xlim(1e-3, 1e2)
plt.ylim(1e-3, 1e2)

#%% 
# # Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# # directory to store the fitted results
# regression_results_dir, weighted = results_output_dir + f'/regression_results_smf_{min_channel}_channel_at_least', 'ols'
# regression_results_dir, weighted = results_output_dir + f'/regression_results_smf_weighted_{min_channel}_channel_at_least', 'wls'

# mkdir(regression_results_dir)

# for nearby_channel_number in combined_channel_number_list:
#     # Load the processed DataFrame
#     peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
#     # Specify magnitude range to do regression
#     snr_threshold = 10

#     # regression
#     results_file_name = f"regression_combined_site_terms_{nearby_channel_number}chan"
#     regP, regS = fit_regression(peak_amplitude_df, weighted=weighted, 
#                                 M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)
    
#     # store the regression results
#     store_regression_results(regP, regression_results_dir, results_filename=f"/P_{results_file_name}")
#     store_regression_results(regS, regression_results_dir, results_filename=f"/S_{results_file_name}")

#     #add secondary calibration second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df)
#     second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df)

#     # extract the site terms from regression results
#     site_term_df = extract_site_terms(regP, regS, peak_amplitude_df)
#     site_term_df = pd.merge(site_term_df, second_calibration, on=['region_site', 'channel_id'])
#     site_term_df.to_csv(regression_results_dir + f'/site_terms_{nearby_channel_number}chan_calibrated.csv', index=False)

