#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import filter_event
from utility.regression import predict_magnitude, get_mean_magnitude
from utility.plotting import plot_magnitude_seaborn

def combine_results(peak_amplitude_df_temp, magnitude_X, wave_type):
    wave_type = wave_type.upper()
    individual_magnitude = peak_amplitude_df_temp[['event_id', 'channel_id', 'magnitude', f'snr{wave_type}']]
    individual_magnitude = individual_magnitude.rename(columns={f'snr{wave_type}': 'snr'})
    individual_magnitude['predicted_magnitude'] = magnitude_X
    individual_magnitude['magnitude_error'] = individual_magnitude['predicted_magnitude'] - individual_magnitude['magnitude']
    return individual_magnitude
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
vmax_list = []
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
for ii in range(len(peak_file_name_list)):
    
    peak_file_name = peak_file_name_list[ii]
    regression_results_dir = regression_results_dir_list[ii]
    result_label = result_label_list[ii]
    snr_threshold = snr_threshold_list[ii]
    M_threshold = M_threshold_list[ii]
    vmax = vmax_list[ii]
    region_text = region_text_list[ii]
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
        individual_magnitude_P = combine_results(peak_amplitude_df_temp, magnitude_P, 'P')
        final_magnitude_P = get_mean_magnitude(peak_amplitude_df_temp, magnitude_P)
    except:
        print('No P regression results, skip...')
        regP, magnitude_P, peak_amplitude_df_temp, final_magnitude_P = None, None, None, None

    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{result_label}.pickle")
        magnitude_S, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regS, site_term_df, wavetype='S')
        individual_magnitude_S = combine_results(peak_amplitude_df_temp, magnitude_S, 'S')
        final_magnitude_S = get_mean_magnitude(peak_amplitude_df_temp, magnitude_S)
    except:
        print('No S regression results, skip...')
        regS, magnitude_S, peak_amplitude_df_temp, final_magnitude_S = None, None, None, None


    # ==========plot figures of strain rate validation
    fig_dir = regression_results_dir + '/figures'
    mkdir(fig_dir)

    xy_lim, height, space = [-1, 8], 10, 0.3


    temp = np.load(regression_results_dir + '/transfer_event_list.npz')
    event_id_fit_P = temp['event_id_fit_P']
    event_id_fit_S = temp['event_id_fit_S']
    event_id_predict = temp['event_id_predict_P']
    event_id_predict = np.array([9001, 9006])

    try:
        final_magnitude_P_fit = final_magnitude_P[final_magnitude_P.event_id.isin(event_id_fit_P)]
        final_magnitude_P_predict = final_magnitude_P[final_magnitude_P.event_id.isin(event_id_predict)]
    
        gP = plot_magnitude_seaborn(final_magnitude_P_predict, xlim=xy_lim, ylim=xy_lim, vmax=vmax[0], height=height, space=space)
        gP.ax_joint.plot(final_magnitude_P_fit.magnitude, final_magnitude_P_fit.predicted_M, 'ro')
        gP.ax_joint.plot(final_magnitude_P_predict.magnitude, final_magnitude_P_predict.predicted_M, 'bo')
        gP.ax_joint.text(-0.5, 7, region_text + f', P wave\n{len(final_magnitude_P_fit.dropna())} events to fit, {len(final_magnitude_P_predict.dropna())} events to predict', fontsize=16)
        gP.savefig(fig_dir + f'/P_magnitude_prediction_rate_{result_label}_seaborn.png')
    except:
        print('No valid P wave regression results, skip ...')
        pass

    try:
        final_magnitude_S_fit = final_magnitude_S[final_magnitude_S.event_id.isin(event_id_fit_S)]
        final_magnitude_S_predict = final_magnitude_S[final_magnitude_S.event_id.isin(event_id_predict)]

        gS = plot_magnitude_seaborn(final_magnitude_S_predict, xlim=xy_lim, ylim=xy_lim, vmax=vmax[1], height=height, space=space)
        gS.ax_joint.plot(final_magnitude_S_fit.magnitude, final_magnitude_S_fit.predicted_M, 'ro')
        gS.ax_joint.plot(final_magnitude_S_predict.magnitude, final_magnitude_S_predict.predicted_M, 'bo')
        gS.ax_joint.text(-0.5, 7, region_text + f', S wave\n{len(final_magnitude_S_fit.dropna())} events to fit, {len(final_magnitude_S_predict.dropna())} events to predict', fontsize=16)
        gS.savefig(fig_dir + f'/S_magnitude_prediction_rate_{result_label}_seaborn.png')
    except:
        print('No valid S wave regression results, skip ...')
        pass


#%%
fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
gca = ax[0]
sns.violinplot(data=individual_magnitude_P, x="event_id", y="magnitude_error", scale='count', bw=0.1, width=1, ax=gca)
gca.set_ylabel('Magnitude P error')
gca.plot([-1, 3], [0, 0], '-k', zorder=-1)
gca.text(-0.7, 1.15, f'Catalog M: {final_magnitude_P_predict.iloc[0, 1]}\nEstimated M: {final_magnitude_P_predict.iloc[0, 2]:.1f}',fontsize=15)
gca.text(0.3,  1.15, f'Catalog M: {final_magnitude_P_predict.iloc[1, 1]}\nEstimated M: {final_magnitude_P_predict.iloc[1, 2]:.1f}',fontsize=15)
gca.set_xlabel('')


event_names = ['60 km W of La Ligua', '40 km NW of Valparaíso', '']
gca = ax[1]
sns.violinplot(data=individual_magnitude_S, x="event_id", y="magnitude_error", scale='count',  bw=0.1, ax=gca)
gca.set_ylabel('Magnitude S error')
gca.plot([-1, 3], [0, 0], '-k', zorder=-1)
gca.text(-0.7, 1.15, f'Catalog M: {final_magnitude_S_predict.iloc[0, 1]}\nEstimated M: {final_magnitude_S_predict.iloc[0, 2]:.1f}',fontsize=15)
gca.text(0.3,  1.15, f'Catalog M: {final_magnitude_S_predict.iloc[1, 1]}\nEstimated M: {final_magnitude_S_predict.iloc[1, 2]:.1f}',fontsize=15)
gca.set_xlim(-1, 2)
gca.set_xlabel('')
gca.set_xticklabels(event_names, rotation=15)

plt.savefig(fig_dir + f'/magnitude_error_violin_{result_label}.png', bbox_inches='tight')
# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(site_term_df.channel_id, site_term_df.site_term_P, '.', label='P')
ax.plot(site_term_df.channel_id, site_term_df.site_term_S, '.', label='S')
ax.legend(loc=4)
ax.set_xlabel('Channel number')
ax.set_ylabel('Site term (log10)')
plt.savefig(fig_dir + f'/site_terms_{result_label}.png', bbox_inches='tight')
# %%
