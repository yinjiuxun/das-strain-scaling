# %%
# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import remove_outliers
from utility.regression import store_regression_results, fit_regression_iteration

def main(data_file, results_output_dir, weighted='wls', wavetypes=['P', 'S'], 
         snr_threshold=10, min_channel=100, n_iter=20, rms_epsilon=0.1):
    
    # Set parameters
    outlier_value = 1e4

    # Create output directory
    mkdir(results_output_dir)

    # Load data
    peak_amplitude_dataframe = load_peak_amplitude_data(data_file, outlier_value)

    # Fit regressions
    regression_results = {}
    site_terms_dataframe = pd.DataFrame(columns=['region', 'channel_id', 'site_term_P', 'site_term_S'])
    for wavetype in wavetypes:
        try:
            regression, site_terms = fit_regression_iteration(peak_amplitude_dataframe, wavetype=wavetype, weighted=weighted,
                                                               M_threshold=None, snr_threshold=snr_threshold,
                                                               min_channel=min_channel, n_iter=n_iter, rms_epsilon=rms_epsilon)
            regression_results[wavetype] = regression
            site_terms['wavetype'] = wavetype
            # site_terms_dataframe = site_terms_dataframe.append(site_terms, ignore_index=True)
            site_terms_dataframe = pd.concat((site_terms_dataframe, site_terms), ignore_index=True)
        except Exception as e:
            print(f"{wavetype} regression failed with error: {e}")

    # Store results
    store_regression_results(regression_results['P'], results_output_dir, results_filename="/P_regression_combined_site_terms_iter")
    store_regression_results(regression_results['S'], results_output_dir, results_filename="/S_regression_combined_site_terms_iter")
    site_terms_dataframe.to_csv(f"{results_output_dir}/site_terms_iter.csv", index=False)


def load_peak_amplitude_data(data_file, outlier_value):
    with open(f"{data_file}", 'r') as f:
        peak_amplitude_dataframe = pd.read_csv(f)

    # use hypocentral distance instead of epicentral distance
    peak_amplitude_dataframe['distance_in_km'] = peak_amplitude_dataframe['calibrated_distance_in_km']

    # remove columns not needed
    peak_amplitude_dataframe = peak_amplitude_dataframe[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km',
                                                         'snrP', 'snrS', 'peak_P', 'peak_S', 'region']]
    
    # remove some outliers
    peak_amplitude_dataframe = remove_outliers(peak_amplitude_dataframe, outlier_value=outlier_value)
    peak_amplitude_dataframe = peak_amplitude_dataframe.drop(index=peak_amplitude_dataframe[peak_amplitude_dataframe.peak_P <= 0].index)
    peak_amplitude_dataframe = peak_amplitude_dataframe.drop(index=peak_amplitude_dataframe[peak_amplitude_dataframe.peak_S <= 0].index)

    return peak_amplitude_dataframe

if __name__ == '__main__':
    # setup the directory of data and results
    data_file = '../data_files/peak_amplitude/peak_amplitude_multiple_arrays.csv'
    results_output_dir = '../iter_results'
    mkdir(results_output_dir)

    # run the iterative regression
    main(data_file, results_output_dir)

# # %%
# #%% import modules
# import pandas as pd
# #from sep_util import read_file
# import numpy as np
# import matplotlib.pyplot as plt

# # import utility functions
# import sys
# sys.path.append('../')
# from utility.general import mkdir
# from utility.processing import combined_regions_for_regression, filter_event, remove_outliers
# from utility.regression import store_regression_results, fit_regression_iteration

# #%% 
# # some parameters
# snr_threshold = 10
# min_channel = 100 # do regression only on events recorded by at least 100 channels

# # parameters for the iterative regression
# n_iter = 20
# rms_epsilon = 0.1 # percentage of rms improvement, if smaller, stop iteration

# # data path
# data_dir = '../data_files/peak_amplitude'

# # result directory
# results_output_dir = '../results'
# mkdir(results_output_dir)

# #%% Iteratively fitting
# weighted = 'wls' # 'ols' or 'wls'
# if weighted == 'ols':
#     weight_text = '' 
# elif weighted == 'wls':
#     weight_text = '_weighted' 
# else:
#     raise

# regression_results_dir = results_output_dir + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least_std'
# mkdir(regression_results_dir)

# peak_amplitude_df0 = pd.read_csv(data_dir + '/peak_amplitude_multiple_arrays.csv')
# peak_amplitude_df0['distance_in_km'] = peak_amplitude_df0['calibrated_distance_in_km']

# peak_amplitude_df0 = peak_amplitude_df0[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km', 
#                                     'snrP', 'snrS', 'peak_P', 'peak_S', 'region']] 

# # to remove some extreme values
# peak_amplitude_df0 = remove_outliers(peak_amplitude_df0, outlier_value=1e4)
# peak_amplitude_df0 = peak_amplitude_df0.drop(index=peak_amplitude_df0[peak_amplitude_df0.peak_P<=0].index)
# peak_amplitude_df0 = peak_amplitude_df0.drop(index=peak_amplitude_df0[peak_amplitude_df0.peak_S<=0].index)



# try:
#     regP, site_term_df_P = fit_regression_iteration(peak_amplitude_df0, wavetype='P', weighted=weighted, 
#                                 M_threshold=None, snr_threshold=snr_threshold, min_channel=min_channel, 
#                                 n_iter=n_iter, rms_epsilon=rms_epsilon)
# except:
#     print('P regression is unavailable, assign Nan and None')
#     regP = None
#     site_term_df_P = pd.DataFrame(columns=['region', 'channel_id', 'site_term_P'])

# try:
#     regS, site_term_df_S = fit_regression_iteration(peak_amplitude_df0, wavetype='S', weighted=weighted, 
#                                 M_threshold=None, snr_threshold=snr_threshold, min_channel=min_channel, 
#                                 n_iter=n_iter, rms_epsilon=rms_epsilon)
# except:
#     print('S regression is unavailable, assign Nan and None')
#     regS = None
#     site_term_df_S = pd.DataFrame(columns=['region', 'channel_id', 'site_term_S'])

# # merge the site term
# site_term_df = pd.merge(site_term_df_P, site_term_df_S, how='outer', left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])

# # store the regression results
# results_file_name = "regression_combined_site_terms_iter"
# store_regression_results(regP, regression_results_dir, results_filename=f"/P_{results_file_name}")
# store_regression_results(regS, regression_results_dir, results_filename=f"/S_{results_file_name}")
# site_term_df.to_csv(regression_results_dir + f'/site_terms_iter.csv', index=False)




# #%% 
# # Apply to single array
# region_list = ['ridgecrest', 'mammothS', 'mammothN', 'sanriku']
# results_output_dirs = ["/OUTPUT/PATH/TO/Ridgecrest",
#                        "/OUTPUT/PATH/TO/LV_S",
#                        "/OUTPUT/PATH/TO/LV_N", 
#                        "/OUTPUT/PATH/TO/Sanriku"]

# M_threshold_list = [[2, 10], [2, 10], [2, 10], [2, 10]]
# snr_threshold_list = [10, 10, 10, 5]
# min_channel = 100

# weighted = 'wls' # 'ols' or 'wls'
# if weighted == 'ols':
#     weight_text = '' 
# elif weighted == 'wls':
#     weight_text = '_weighted' 
# else:
#     raise

# for i_region in range(len(results_output_dirs)):
#     M_threshold = M_threshold_list[i_region]
#     snr_threshold = snr_threshold_list[i_region]
#     results_output_dir = results_output_dirs[i_region]
#     print(results_output_dir)

#     regression_results_dir = results_output_dir + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
#     mkdir(regression_results_dir)

#     peak_amplitude_df = peak_amplitude_df0[peak_amplitude_df0.region == region_list[i_region]]

#     if region_list[i_region] == 'sanriku': # some special processing for Sanriku data
#         peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.QA == 'Yes']
#         peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 4130].index)
#         #peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 1580].index)

#     peak_amplitude_df = peak_amplitude_df[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km', 
#                                         'snrP', 'snrS', 'peak_P', 'peak_S', 'region']] 
                                    
#     # to remove some extreme values
#     peak_amplitude_df = remove_outliers(peak_amplitude_df, outlier_value=1e4)
#     peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.peak_P<=0].index)
#     peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.peak_S<=0].index)

#     # Remove some bad data (clipped, poor-quality)
#     if region_list[i_region] == 'ridgecrest':
#         peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id.isin([38548295.0, 39462536.0])].index)
#     if region_list[i_region] == 'mammothN':
#         peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 73584926].index)
#     if region_list[i_region] == 'mammothS':
#         peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 73584926].index)

#     n_iter = 50
#     rms_epsilon = 0.1 # percentage of rms improvement, if smaller, stop iteration
#     show_figure = True
    
#     try:
#         regP, site_term_df_P = fit_regression_iteration(peak_amplitude_df, wavetype='P', weighted=weighted, 
#                                     M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel, 
#                                     n_iter=n_iter, rms_epsilon=rms_epsilon, show_figure=show_figure)
#     except:
#         print('P regression is unavailable, assign Nan and None')
#         regP = None
#         site_term_df_P = pd.DataFrame(columns=['region', 'channel_id', 'site_term_P', 'site_term_P_STD'])

#     try:
#         regS, site_term_df_S = fit_regression_iteration(peak_amplitude_df, wavetype='S', weighted=weighted, 
#                                     M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel, 
#                                     n_iter=n_iter, rms_epsilon=rms_epsilon, show_figure=show_figure)
#     except:
#         print('S regression is unavailable, assign Nan and None')
#         regS = None
#         site_term_df_S = pd.DataFrame(columns=['region', 'channel_id', 'site_term_S', 'site_term_S_STD'])

#     # merge the site term
#     site_term_df = pd.merge(site_term_df_P, site_term_df_S, how='outer', left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])

#     # store the regression results
#     results_file_name = "regression_combined_site_terms_iter"
#     store_regression_results(regP, regression_results_dir, results_filename=f"/P_{results_file_name}")
#     store_regression_results(regS, regression_results_dir, results_filename=f"/S_{results_file_name}")
#     site_term_df.to_csv(regression_results_dir + f'/site_terms_iter.csv', index=False)
# %%
