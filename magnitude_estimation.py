#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt

from utility.general import mkdir
from utility.processing import filter_event
from utility.regression import predict_magnitude, get_mean_magnitude
from utility.plotting import plot_magnitude_seaborn

#%%
# some parameters
snr_threshold = 10
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [2.5, 10]

results_output_dir_list = []
peak_file_name_list = []

# # Set result directory
# multiple arrays
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
peak_file_name = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/peak_amplitude_multiple_arrays.csv'
results_output_dir_list.append(results_output_dir)
peak_file_name_list.append(peak_file_name)

# single arrays
# Ridgecrest
results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
peak_file_name = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
results_output_dir_list.append(results_output_dir)
peak_file_name_list.append(peak_file_name)

# Long Valley N
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
peak_file_name = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events/calibrated_peak_amplitude.csv'
results_output_dir_list.append(results_output_dir)
peak_file_name_list.append(peak_file_name)

# Long Valley S
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
peak_file_name = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events/calibrated_peak_amplitude.csv'
results_output_dir_list.append(results_output_dir)
peak_file_name_list.append(peak_file_name)

#%%
for weight_text in ['_weighted', '']:
    for (results_output_dir, peak_file_name) in zip(results_output_dir_list, peak_file_name_list):
        print(results_output_dir)
        # load results
        regression_results_dir = results_output_dir + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least'

        peak_amplitude_df = pd.read_csv(peak_file_name)
        peak_amplitude_df = filter_event(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)

        regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_iter.pickle")
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_iter.pickle")
        site_term_df = pd.read_csv(regression_results_dir + '/site_terms_iter.csv')

        # use the measured peak amplitude to estimate the magnitude
        magnitude_P, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regP, site_term_df, wavetype='P')
        magnitude_S, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regS, site_term_df, wavetype='S')

        final_magnitude_P = get_mean_magnitude(peak_amplitude_df_temp, magnitude_P)
        final_magnitude_S = get_mean_magnitude(peak_amplitude_df_temp, magnitude_S)

        # check the STD of magnitude estimation, if too large, discard
        fig, ax = plt.subplots()
        ax.hist(final_magnitude_P.predicted_M_std, label='P')
        ax.hist(final_magnitude_S.predicted_M_std, label='S')
        ax.legend()

        # final_magnitude_P = final_magnitude_P[final_magnitude_P.predicted_M_std < 0.75]
        # final_magnitude_S = final_magnitude_S[final_magnitude_S.predicted_M_std < 0.75]

        # plot figures of strain rate validation
        fig_dir = regression_results_dir + '/figures'
        mkdir(fig_dir)

        xy_lim, height, space = [-1, 8], 10, 0.3

        g = plot_magnitude_seaborn(final_magnitude_P, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
        g.savefig(fig_dir + f'/P_magnitude_prediction_rate_iter_seaborn.png')

        g = plot_magnitude_seaborn(final_magnitude_S, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
        g.savefig(fig_dir + f'/S_magnitude_prediction_rate_iter_seaborn.png')

# %%
