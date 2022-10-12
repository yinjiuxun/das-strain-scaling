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
# results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
# peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
# result_label = 'transfer'
# snr_threshold = 10
# M_threshold = [2, 10]

results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
peak_file_name = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
result_label = 'transfer'
snr_threshold = 5
M_threshold = [2, 10]
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

regression_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least'

#  ================== transfered test ================== 
N_event_fit_list = range(2, 30)
N_test = 50

good_ratio_all_P = np.zeros(shape=(len(N_event_fit_list), N_test))
good_ratio_all_S = np.zeros(shape=(len(N_event_fit_list), N_test))

for i_fit, N_event_fit in enumerate(N_event_fit_list):
    print(f'=============== {N_event_fit} events to fit ===============')
    good_ratio_test_P = np.zeros(N_test)
    good_ratio_test_S = np.zeros(N_test)
    for i_test in range(N_test):
        if i_test%10 == 0:
            print(f'=============== {i_test} th test ===============')

        regression_results_dir = regression_dir + f'/{N_event_fit}_fit_events_{i_test}th_test'

        # load results
        peak_amplitude_df = pd.read_csv(peak_file_name)
        peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
        peak_amplitude_df = filter_event(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)

        site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{result_label}.csv')

        try:
            regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{result_label}.pickle")
            # use the measured peak amplitude to estimate the magnitude
            magnitude_P, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regP, site_term_df, wavetype='P')
            final_magnitude_P = get_mean_magnitude(peak_amplitude_df_temp, magnitude_P)
            mag_err_P = final_magnitude_P.predicted_M - final_magnitude_P.magnitude

            good_ratio_P = len(mag_err_P[abs(mag_err_P) < 0.5]) / len(mag_err_P[~mag_err_P.isna()])
            good_ratio_test_P[i_test] = good_ratio_P

        except:
            #print('No P regression results, skip...')
            good_ratio_test_P[i_test] = np.nan

        try:
            regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{result_label}.pickle")
            magnitude_S, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regS, site_term_df, wavetype='S')
            final_magnitude_S = get_mean_magnitude(peak_amplitude_df_temp, magnitude_S)
            mag_err_S = final_magnitude_S.predicted_M - final_magnitude_S.magnitude

            good_ratio_S = len(mag_err_S[abs(mag_err_S) < 0.5]) / len(mag_err_S[~mag_err_S.isna()])
            good_ratio_test_S[i_test] = good_ratio_S

        except:
            #print('No S regression results, skip...')
            good_ratio_test_S[i_test] = np.nan

    good_ratio_all_P[i_fit, :] = good_ratio_test_P
    good_ratio_all_S[i_fit, :] = good_ratio_test_S

np.savez(results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/good_magnitude_ratio.npz', 
         good_ratio_all_P=good_ratio_all_P, good_ratio_all_S=good_ratio_all_S, N_event_fit_list=N_event_fit_list)
# %%
regression_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least'
temp_read = np.load(regression_dir +'/good_magnitude_ratio.npz')

good_ratio_all_P = temp_read['good_ratio_all_P']*100
good_ratio_all_S = temp_read['good_ratio_all_S']*100
N_event_fit_list = temp_read['N_event_fit_list']

fig, gca = plt.subplots(figsize=(8, 5))
gca.plot(N_event_fit_list, good_ratio_all_S, 'k.', alpha=0.2)
gca.plot(N_event_fit_list, np.nanmedian(good_ratio_all_S, axis=1), '-r', label='median')
gca.legend()
gca.set_xlabel('# of event to fit')
gca.set_ylabel('% of good estimation')
plt.savefig(regression_dir + '/good_magnitude_ratio.png', bbox_inches='tight')
plt.savefig(regression_dir + '/good_magnitude_ratio.pdf', bbox_inches='tight')
# %%
