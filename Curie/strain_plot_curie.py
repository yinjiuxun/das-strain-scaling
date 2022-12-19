#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import combined_channels, filter_event, get_comparison_df
from utility.regression import predict_strain
from utility.plotting import plot_prediction_vs_measure_seaborn

#%%
# some parameters
min_channel = 100 # do regression only on events recorded by at least 100 channels
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
snr_threshold_list = []
vmax_list = []
M_threshold_list = []
region_text_list = []


#  ================== Curie transfered specified test ================== 
results_output_dir = '/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate'
peak_file_name = '/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
result_label = 'transfer'
regression_results_dir = results_output_dir + f'/transfer_regression_specified_smf{weight_text}_{min_channel}_channel_at_least_9007/'
snr_threshold = 10
vmax = 50
M_threshold = [2, 10]
vmax = [2, 2] # for P and S
region_text = 'Transfered scaling for Curie'

M_threshold_list.append(M_threshold)
results_output_dir_list.append(results_output_dir)
regression_results_dir_list.append(regression_results_dir)
peak_file_name_list.append(peak_file_name)
result_label_list.append(result_label)
snr_threshold_list.append(snr_threshold)
vmax_list.append(vmax)
region_text_list.append(region_text) 


#%%
# Validating strain rate TODO: correct here!
for ii in range(len(peak_file_name_list)):
    
    peak_file_name = peak_file_name_list[ii]
    regression_results_dir = regression_results_dir_list[ii]
    result_label = result_label_list[ii]
    snr_threshold = snr_threshold_list[ii]
    region_text = region_text_list[ii]
    print(regression_results_dir)

    # load results
    peak_amplitude_df = pd.read_csv(peak_file_name)
    peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']

    peak_amplitude_df = filter_event(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)

    site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{result_label}.csv')

    try:
        regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{result_label}.pickle")
        peak_P_predicted, peak_amplitude_df_temp = predict_strain(peak_amplitude_df, regP, site_term_df, wavetype='P')
        peak_P_comparison_df = get_comparison_df(data=[peak_amplitude_df_temp.event_id, peak_amplitude_df_temp.peak_P, peak_P_predicted], 
                                            columns=['event_id', 'peak_P', 'peak_P_predict'])
    except:
        print('No P regression results, skip...')
        regP, peak_P_predicted, peak_amplitude_df_temp, peak_P_comparison_df = None, None, None, None


    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{result_label}.pickle")
        peak_S_predicted, peak_amplitude_df_temp = predict_strain(peak_amplitude_df, regS, site_term_df, wavetype='S')
        peak_S_comparison_df = get_comparison_df(data=[peak_amplitude_df_temp.event_id, peak_amplitude_df_temp.peak_S, peak_S_predicted], 
                                            columns=['event_id', 'peak_S', 'peak_S_predict'])
    except:
        print('No S regression results, skip...')
        regS, peak_S_predicted, peak_amplitude_df_temp, peak_S_comparison_df = None, None, None, None


    # plot figures of strain rate validation
    fig_dir = regression_results_dir + '/figures'
    mkdir(fig_dir)

    xy_lim, height, space = [1e-3, 1e2], 10, 0.3
    vmax = vmax_list[ii]

    if result_label == 'iter': 
        try:
            gP = plot_prediction_vs_measure_seaborn(peak_P_comparison_df, phase='P', bins=40, vmax=vmax, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gP.ax_joint.text(2e-3, 50, region_text + f', P wave\n{len(peak_P_comparison_df.dropna())} measurements')
            gP.savefig(fig_dir + f'/P_validate_strain_rate_{result_label}_seaborn.png')
        except:
            print('No P regression results, no figure, skip...')

        try:
            gS = plot_prediction_vs_measure_seaborn(peak_S_comparison_df, phase='S', bins=40, vmax=vmax, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gS.ax_joint.text(2e-3, 50, region_text + f', S wave\n{len(peak_S_comparison_df.dropna())} measurements')
            gS.savefig(fig_dir + f'/S_validate_strain_rate_{result_label}_seaborn.png')
        except:
            print('No S regression results, no figure, skip...')
        
    elif result_label == 'transfer':
        temp = np.load(regression_results_dir + '/transfer_event_list.npz')
        event_id_fit_P = temp['event_id_fit_P']
        event_id_fit_S = temp['event_id_fit_S']
        event_id_predict = temp['event_id_predict_P']

        try:
            final_peak_fit = peak_P_comparison_df[peak_P_comparison_df.event_id.isin(event_id_fit_P)]
            final_peak_predict = peak_P_comparison_df[peak_P_comparison_df.event_id.isin(event_id_predict)]

            gP = plot_prediction_vs_measure_seaborn(final_peak_predict, phase='P', bins=40, vmax=vmax, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gP.ax_joint.loglog(final_peak_fit.peak_P, final_peak_fit.peak_P_predict, 'r.')
            gP.ax_joint.text(2e-3, 50, region_text + f', P wave\n{len(final_peak_fit.dropna())} measurments to fit, \n{len(final_peak_predict.dropna())} measurments to predict', fontsize=15)
            gP.savefig(fig_dir + f'/P_validate_strain_rate_{result_label}_seaborn.png')
        except:
            print('No valid P wave regression results, skip ...')
            pass

        try:
            final_peak_fit = peak_S_comparison_df[peak_S_comparison_df.event_id.isin(event_id_fit_S)]
            final_peak_predict = peak_S_comparison_df[peak_S_comparison_df.event_id.isin(event_id_predict)]
        
            gS = plot_prediction_vs_measure_seaborn(final_peak_predict, phase='S', bins=40, vmax=vmax, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gS.ax_joint.loglog(final_peak_fit.peak_S, final_peak_fit.peak_S_predict, 'r.')
            gS.ax_joint.text(2e-3, 50, region_text + f', S wave\n{len(final_peak_fit.dropna())} measurments to fit, \n{len(final_peak_predict.dropna())} measurments to predict', fontsize=15)
            gS.savefig(fig_dir + f'/S_validate_strain_rate_{result_label}_seaborn.png')
        except:
            print('No valid P wave regression results, skip ...')
            pass

    plt.close('all')
    


fig, ax = plt.subplots(2,1, figsize=(10, 10))

temp_df0 = peak_P_comparison_df[peak_P_comparison_df.event_id==9007]
temp_df1 = peak_P_comparison_df[peak_P_comparison_df.event_id==9006]
temp_df2 = peak_P_comparison_df[peak_P_comparison_df.event_id==9001]
ax[0].loglog(temp_df0.peak_P_predict, temp_df0.peak_P, '.', alpha=1)
ax[0].loglog(temp_df1.peak_P_predict, temp_df1.peak_P, '.', alpha=0.5)
ax[0].loglog(temp_df2.peak_P_predict, temp_df2.peak_P, '.', alpha=0.5)

temp_df0 = peak_S_comparison_df[peak_S_comparison_df.event_id==9007]
temp_df1 = peak_S_comparison_df[peak_S_comparison_df.event_id==9006]
temp_df2 = peak_S_comparison_df[peak_S_comparison_df.event_id==9001]
ax[1].loglog(temp_df0.peak_S_predict, temp_df0.peak_S, '.', alpha=1)
ax[1].loglog(temp_df1.peak_S_predict, temp_df1.peak_S, '.', alpha=0.5)
ax[1].loglog(temp_df2.peak_S_predict, temp_df2.peak_S, '.', alpha=0.5)

#%%
fig, ax = plt.subplots(2,1, figsize=(10, 10))
gca = ax[0]
peak_P_comparison_df['error'] = np.log10(abs(peak_P_comparison_df['peak_P']/(peak_P_comparison_df['peak_P_predict'])))
peak_P_comparison_df = peak_P_comparison_df.dropna()
sns.violinplot(data=peak_P_comparison_df, x="event_id", y="error", scale='count', ax=gca)

gca = ax[1]
peak_S_comparison_df['error'] = np.log10(abs(peak_S_comparison_df['peak_S']/(peak_S_comparison_df['peak_S_predict'])))
peak_S_comparison_df = peak_S_comparison_df.dropna()
sns.violinplot(data=peak_S_comparison_df, x="event_id", y="error", scale='count', ax=gca)
# %%
