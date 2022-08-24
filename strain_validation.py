#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# import the plotting functions
from plotting_functions import *
from utility_functions import *

#%% Functions used here
def filter_by_channel_number(peak_amplitude_df, min_channel):
    """To remove the measurements from few channels (< min_channel)"""
    event_channel_count = peak_amplitude_df.groupby(['event_id'])['event_id'].count()
    channel_count = event_channel_count.values
    event_id = event_channel_count.index
    event_id = event_id[channel_count >= min_channel]

    return peak_amplitude_df[peak_amplitude_df['event_id'].isin(event_id)]


def combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number):
    if nearby_channel_number == -1:
        peak_amplitude_df['combined_channel_id'] = 0
    else:
        temp1= np.arange(0, DAS_index.max()+1) # original channel number
        temp2 = temp1 // nearby_channel_number # combined channel number
        peak_amplitude_df['combined_channel_id'] = temp2[np.array(peak_amplitude_df.channel_id).astype('int')]
    return peak_amplitude_df

def process_region_param_keys(reg, region_string):
    '''A special function to process the regional keys'''
    temp0 = reg.params.keys()
    temp = [region_string in tempx for tempx in temp0]
    site_term_keys = temp0[temp]

    combined_channel = np.array([int(k.replace(f'C(region_site)[{region_string}-','').replace(']','')) for k in site_term_keys])
    ii_sort = np.argsort(combined_channel)

    combined_channel = combined_channel[ii_sort]
    site_term = reg.params[temp].values[ii_sort]

    return combined_channel, site_term

import seaborn as sns
def plot_prediction_vs_measure_seaborn(peak_comparison_df, xy_range, phase, bins=40, vmin=None, vmax=None):
    sns.set_theme(style="ticks", font_scale=2)
    if phase == 'P':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_P", y="peak_P_predict", marginal_ticks=True,
                        xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    elif phase == 'S':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_S", y="peak_S_predict", marginal_ticks=True,
                        xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    g.ax_joint.set(xscale="log")
    g.ax_joint.set(yscale="log")

# Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

# Add the joint and marginal histogram plots 03012d
    g.plot_joint(sns.histplot, discrete=(False, False), bins=(bins*2,bins), cmap="light:#4D4D9C", vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", bins=bins, color="#4D4D9C")

    g.ax_joint.plot(xy_range, xy_range, 'k-', linewidth = 2)
    g.ax_joint.set_xlabel('measured peak strain rate\n (micro strain/s)')
    g.ax_joint.set_ylabel('calculated peak strain rate\n (micro strain/s)')
    return g
# ==============================  Ridgecrest data ========================================
#%% Specify the file names
# results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
# das_pick_file_name = '/peak_amplitude_M3+.csv'
# region_label = 'ridgecrest'

results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events'
das_pick_file_name = '/calibrated_peak_amplitude.csv'
region_label = 'ridgecrest'

# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling'
das_pick_file_name = '/peak_amplitude_M3+.csv'
region_label = 'olancha'

# ==============================  Mammoth data - South========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events'
das_pick_file_name = '/calibrated_peak_amplitude.csv'
region_label = 'mammothS'

# ==============================  Mammoth data - North========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events'
das_pick_file_name = '/calibrated_peak_amplitude.csv'
region_label = 'mammothN'

# %% load the peak amplitude results
# Load the peak amplitude results
min_channel = 100 # do regression only on events recorded by at least 100 channels
snr_threshold = 10
magnitude_threshold = [2, 10]
apply_calibrated_distance = True # if true, use the depth-calibrated distance to do regression

peak_amplitude_df = pd.read_csv(das_pick_file_folder + '/' + das_pick_file_name)

if apply_calibrated_distance: 
    peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']

# directory to store the fitted results
regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.snrP >=snr_threshold) | (peak_amplitude_df.snrS >=snr_threshold)]
peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >=magnitude_threshold[0]) & (peak_amplitude_df.magnitude <=magnitude_threshold[1])]

peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.peak_P>0]
peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.peak_S>0]

#peak_amplitude_df = peak_amplitude_df.dropna()
DAS_index = peak_amplitude_df.channel_id.unique().astype('int')


# %% Compare the regression parameters and site terms
fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
# DataFrame to store parameters for all models
P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'], 
index = np.arange(len(combined_channel_number_list)))
S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
index = np.arange(len(combined_channel_number_list)))

for i_model, combined_channel_number in enumerate(combined_channel_number_list):

    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")

    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)
    combined_channel_id = np.sort(peak_amplitude_df.combined_channel_id.unique())
    peak_amplitude_df = add_event_label(peak_amplitude_df)

    y_P_predict = regP.predict(peak_amplitude_df)
    y_S_predict = regS.predict(peak_amplitude_df)

    temp_peaks = np.array([np.array(peak_amplitude_df.peak_P),
              np.array(peak_amplitude_df.peak_S), 
              np.array(10**y_P_predict), 
              np.array(10**y_S_predict)]).T
    peak_comparison_df = pd.DataFrame(data=temp_peaks,
                                  columns=['peak_P', 'peak_S', 'peak_P_predict', 'peak_S_predict'])
    
    g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='P')
    g.savefig(regression_results_dir + f'/P_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn.png')

    g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S')
    g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn.png')
    
    plt.close('all')
    # plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict, y_S_predict, (-2.0, 2), 
    # regression_results_dir + f'/validate_predicted__combined_site_terms_{combined_channel_number}chan.png')
    
# %%
# ==============================  Looking into the multiple array case ========================================
# Specify the file names
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'

min_channel = 100
snr_threshold = 10
magnitude_threshold = [2, 10]
apply_calibrated_distance = True # if true, use the depth-calibrated distance to do regression
output_label = ''
apply_secondary_calibration = True # if true, use the secondary site term calibration

# directory to store the fitted results
regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_{min_channel}_channel_at_least'

# %% Compare the regression parameters and site terms
fig, ax = plt.subplots(2, 1, figsize=(10, 18), sharex=True, sharey=True) # north
fig2, ax2 = plt.subplots(2, 1, figsize=(10, 18), sharex=True, sharey=True) # south
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{combined_channel_number}.csv')
    if apply_calibrated_distance: 
        peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']

    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.snrP >=snr_threshold) | (peak_amplitude_df.snrS >=snr_threshold)]
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >=magnitude_threshold[0]) & (peak_amplitude_df.magnitude <=magnitude_threshold[1])]

    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.peak_P>0]
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.peak_S>0]

    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")

    # peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)
    # combined_channel_id = np.sort(peak_amplitude_df.combined_channel_id.unique())
    # peak_amplitude_df = add_event_label(peak_amplitude_df)

    y_P_predict = regP.predict(peak_amplitude_df)
    y_S_predict = regS.predict(peak_amplitude_df)

    if apply_secondary_calibration:
        second_calibration =pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{combined_channel_number}chan.csv')
        peak_amplitude_df = pd.merge(peak_amplitude_df, second_calibration, on=['channel_id', 'region', 'region_site'])
        # apply secondary calibration
        y_P_predict = regP.predict(peak_amplitude_df) + peak_amplitude_df.diff_peak_P
        y_S_predict = regS.predict(peak_amplitude_df) + peak_amplitude_df.diff_peak_S
        output_label = '_calibrated'

    temp_peaks = np.array([np.array(peak_amplitude_df.peak_P),
              np.array(peak_amplitude_df.peak_S), 
              np.array(10**y_P_predict), 
              np.array(10**y_S_predict)]).T
    peak_comparison_df = pd.DataFrame(data=temp_peaks,
                                  columns=['peak_P', 'peak_S', 'peak_P_predict', 'peak_S_predict'])
    
    g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='P', vmax=2e4)
    g.savefig(regression_results_dir + f'/P_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn{output_label}.png')

    g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S', vmax=2e4)
    g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn{output_label}.png')

    plt.close('all')
# %%
# ==============================  Looking into the strain validation for specific channels ========================================
combined_channel_number_list = [100]#[10, 20, 50, 100, -1] # -1 means the constant model

regions_list = ['ridgecrest', 'mammothN', 'mammothS'] #['mammothS']#
channel_lists = [range(0, 1150, 50), range(0, 4000, 200), range(0, 4000, 200)] #[[3000]]#

for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    peak_amplitude_df0 = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{combined_channel_number}.csv')
    if apply_calibrated_distance: 
        peak_amplitude_df0['distance_in_km'] = peak_amplitude_df0['calibrated_distance_in_km']

    peak_amplitude_df0 = peak_amplitude_df0[(peak_amplitude_df0.snrP >=snr_threshold) | (peak_amplitude_df0.snrS >=snr_threshold)]
    peak_amplitude_df0 = peak_amplitude_df0[(peak_amplitude_df0.magnitude >=magnitude_threshold[0]) & (peak_amplitude_df0.magnitude <=magnitude_threshold[1])]

    peak_amplitude_df0 = peak_amplitude_df0[peak_amplitude_df0.peak_P>0]
    peak_amplitude_df0 = peak_amplitude_df0[peak_amplitude_df0.peak_S>0]

    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")

    for ii_region, region_now in enumerate(regions_list):
        for channel in channel_lists[ii_region]:
            # find a specific region and channel
            specific_region_channel = [region_now, channel]
            peak_amplitude_df = peak_amplitude_df0[(peak_amplitude_df0.region==specific_region_channel[0]) & (peak_amplitude_df0.channel_id==specific_region_channel[1])]

            y_P_predict = regP.predict(peak_amplitude_df)
            y_S_predict = regS.predict(peak_amplitude_df)

            temp_peaks = np.array([np.array(peak_amplitude_df.peak_P),
                    np.array(peak_amplitude_df.peak_S), 
                    np.array(10**y_P_predict), 
                    np.array(10**y_S_predict)]).T
            peak_comparison_df = pd.DataFrame(data=temp_peaks,
                                        columns=['peak_P', 'peak_S', 'peak_P_predict', 'peak_S_predict'])
            
            g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='P', bins=20)
            g.savefig(regression_results_dir + f'/channel_validations/P_validation_{specific_region_channel[0]}_channel_{specific_region_channel[1]}.png')

            g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S', bins=20)
            g.savefig(regression_results_dir + f'/channel_validations/S_validation_{specific_region_channel[0]}_channel_{specific_region_channel[1]}.png')

            plt.close('all')


# %%
# %%
# ==============================  Looking into the strain validation for specific channels after secondary calibration ========================================
combined_channel_number_list = [100]#[10, 20, 50, 100, -1] # -1 means the constant model

regions_list = ['ridgecrest', 'mammothN', 'mammothS'] #['mammothS']#
channel_lists = [range(0, 1150, 50), range(0, 4000, 200), range(0, 4000, 200)] #[[3000]]#

for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    peak_amplitude_df0 = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{combined_channel_number}.csv')
    if apply_calibrated_distance: 
        peak_amplitude_df0['distance_in_km'] = peak_amplitude_df0['calibrated_distance_in_km']

    peak_amplitude_df0 = peak_amplitude_df0[(peak_amplitude_df0.snrP >=snr_threshold) | (peak_amplitude_df0.snrS >=snr_threshold)]
    peak_amplitude_df0 = peak_amplitude_df0[(peak_amplitude_df0.magnitude >=magnitude_threshold[0]) & (peak_amplitude_df0.magnitude <=magnitude_threshold[1])]

    peak_amplitude_df0 = peak_amplitude_df0[peak_amplitude_df0.peak_P>0]
    peak_amplitude_df0 = peak_amplitude_df0[peak_amplitude_df0.peak_S>0]

    second_calibration =pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{combined_channel_number}chan.csv')
    peak_amplitude_df0 = pd.merge(peak_amplitude_df0, second_calibration, on=['channel_id', 'region', 'region_site'])

    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    
    for ii_region, region_now in enumerate(regions_list):
        for channel in channel_lists[ii_region]:
            # find a specific region and channel
            specific_region_channel = [region_now, channel]
            peak_amplitude_df = peak_amplitude_df0[(peak_amplitude_df0.region==specific_region_channel[0]) & (peak_amplitude_df0.channel_id==specific_region_channel[1])]

            y_P_predict = regP.predict(peak_amplitude_df)
            y_S_predict = regS.predict(peak_amplitude_df)

            # apply secondary calibration
            y_P_predict = y_P_predict + peak_amplitude_df.diff_peak_P
            y_S_predict = y_S_predict + peak_amplitude_df.diff_peak_S

            temp_peaks = np.array([np.array(peak_amplitude_df.peak_P),
                    np.array(peak_amplitude_df.peak_S), 
                    np.array(10**y_P_predict), 
                    np.array(10**y_S_predict)]).T
            peak_comparison_df = pd.DataFrame(data=temp_peaks,
                                        columns=['peak_P', 'peak_S', 'peak_P_predict', 'peak_S_predict'])
            
            g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='P', bins=20)
            g.savefig(regression_results_dir + f'/channel_validations/P_validation_{specific_region_channel[0]}_channel_{specific_region_channel[1]}_calibrated.png')

            g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S', bins=20)
            g.savefig(regression_results_dir + f'/channel_validations/S_validation_{specific_region_channel[0]}_channel_{specific_region_channel[1]}_calibrated.png')

            plt.close('all')

# %%
