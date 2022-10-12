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
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [2, 10]
weighted = 'ols' # 'ols' or 'wls'
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

#%% # Set result directory
if not random_test:
    # ================== multiple arrays ================== 
    results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
    peak_file_name = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/peak_amplitude_multiple_arrays.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)

    # single arrays
    #  ================== Ridgecrest ================== 
    results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)

    #  ================== Long Valley N ================== 
    results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
    peak_file_name = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)

    #  ================== Long Valley S ================== 
    results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
    peak_file_name = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)

    #  ================== Sanriku fittd ================== 
    results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 5
    M_threshold = [2, 10]

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)

    #  ================== LAX fittd ================== 
    results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)

    # #  ================== LAX transfered ================== 
    # results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
    # peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    # result_label = 'transfer'
    # regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    # snr_threshold = 5
    # M_threshold = [0, 10]

    # M_threshold_list.append(M_threshold)
    # results_output_dir_list.append(results_output_dir)
    # regression_results_dir_list.append(regression_results_dir)
    # peak_file_name_list.append(peak_file_name)
    # result_label_list.append(result_label)
    # snr_threshold_list.append(snr_threshold)

else:

    #  ================== LAX transfered test ================== 
    N_event_fit_list = range(2, 30)
    N_test = 50
    for N_event_fit in N_event_fit_list:
        for i_test in range(N_test):

            results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
            peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
            result_label = 'transfer'
            regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
            snr_threshold = 10
            vmax = 50
            M_threshold = [2, 10]

            M_threshold_list.append(M_threshold)
            results_output_dir_list.append(results_output_dir)
            regression_results_dir_list.append(regression_results_dir)
            peak_file_name_list.append(peak_file_name)
            result_label_list.append(result_label)
            snr_threshold_list.append(snr_threshold)

    # #  ================== Sanriku transfered test ================== 
    # N_event_fit_list = range(2, 30)
    # N_test = 50
    # for N_event_fit in N_event_fit_list:
    #     for i_test in range(N_test):

    #         results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
    #         peak_file_name = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    #         result_label = 'transfer'
    #         regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
    #         snr_threshold = 5
    #         vmax = 50
    #         M_threshold = [2, 10]

    #         M_threshold_list.append(M_threshold)
    #         results_output_dir_list.append(results_output_dir)
    #         regression_results_dir_list.append(regression_results_dir)
    #         peak_file_name_list.append(peak_file_name)
    #         result_label_list.append(result_label)
    #         snr_threshold_list.append(snr_threshold)


#%%
for ii in range(len(peak_file_name_list)):
    
    peak_file_name = peak_file_name_list[ii]
    regression_results_dir = regression_results_dir_list[ii]
    result_label = result_label_list[ii]
    snr_threshold = snr_threshold_list[ii]
    M_threshold = M_threshold_list[ii]
    print(regression_results_dir)

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
    except:
        print('No P regression results, skip...')

    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{result_label}.pickle")
        magnitude_S, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regS, site_term_df, wavetype='S')
        final_magnitude_S = get_mean_magnitude(peak_amplitude_df_temp, magnitude_S)
    except:
        print('No S regression results, skip...')

    # check the STD of magnitude estimation, if too large, discard
    # fig, ax = plt.subplots()
    # ax.hist(final_magnitude_P[~final_magnitude_P.predicted_M_std.isna()].predicted_M_std, label='P')
    # ax.hist(final_magnitude_S[~final_magnitude_S.predicted_M_std.isna()].predicted_M_std, label='S')
    # ax.legend()

    # final_magnitude_P = final_magnitude_P[final_magnitude_P.predicted_M_std < 0.75]
    # final_magnitude_S = final_magnitude_S[final_magnitude_S.predicted_M_std < 0.75]

    # plot figures of strain rate validation
    fig_dir = regression_results_dir + '/figures'
    mkdir(fig_dir)

    xy_lim, height, space = [-1, 8], 10, 0.3

    if result_label == 'iter': 
        try:
            gP = plot_magnitude_seaborn(final_magnitude_P, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gP.savefig(fig_dir + f'/P_magnitude_prediction_rate_{result_label}_seaborn.png')
        except:
            print('No P regression results, skip...')

        try:
            gS = plot_magnitude_seaborn(final_magnitude_S, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gS.savefig(fig_dir + f'/S_magnitude_prediction_rate_{result_label}_seaborn.png')
        except:
            print('No S regression results, skip...')    

    elif result_label == 'transfer':
        temp = np.load(regression_results_dir + '/transfer_event_list.npz')
        event_id_fit = temp['event_id_fit']
        event_id_predict = temp['event_id_predict']

        try:
            final_magnitude_P_fit = final_magnitude_P[final_magnitude_P.event_id.isin(event_id_fit)]
            final_magnitude_P_predict = final_magnitude_P[final_magnitude_P.event_id.isin(event_id_predict)]

            gP = plot_magnitude_seaborn(final_magnitude_P_predict, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gP.ax_joint.plot(final_magnitude_P_fit.magnitude, final_magnitude_P_fit.predicted_M, 'ro')
            gP.savefig(fig_dir + f'/P_magnitude_prediction_rate_{result_label}_seaborn.png')
        except:
            print('No valid P wave regression results, skip ...')
            pass

        try:
            final_magnitude_S_fit = final_magnitude_S[final_magnitude_S.event_id.isin(event_id_fit)]
            final_magnitude_S_predict = final_magnitude_S[final_magnitude_S.event_id.isin(event_id_predict)]

            gS = plot_magnitude_seaborn(final_magnitude_S_predict, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gS.ax_joint.plot(final_magnitude_S_fit.magnitude, final_magnitude_S_fit.predicted_M, 'ro')
            gS.savefig(fig_dir + f'/S_magnitude_prediction_rate_{result_label}_seaborn.png')
        except:
            print('No valid S wave regression results, skip ...')
            pass

    plt.close('all')
# %%
