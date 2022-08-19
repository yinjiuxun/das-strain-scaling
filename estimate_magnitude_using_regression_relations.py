#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# import the plotting functions
from plotting_functions import *
# import the utility functions
from utility_functions import *


# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
# %matplotlib inline
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 18, # fontsize for x and y labels (was 10)
    'axes.titlesize': 18,
    'font.size': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex':False,
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
}
matplotlib.rcParams.update(params)

#%% Define functions
# Use the predicted strain to calculate magnitude
def calculate_magnitude_from_strain(peak_amplitude_df, reg, type, fitting_type='without_site', site_term_column='region_site', secondary_calibration=None):
    if secondary_calibration:
        second_calibration = pd.read_csv(results_output_dir + '/' + regression_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv')
        peak_amplitude_df = pd.merge(peak_amplitude_df, second_calibration, on=['channel_id', 'region', 'region_site'])
        # # apply secondary calibration
        # y_P_predict = y_P_predict + peak_amplitude_df.diff_peak_P
        # y_S_predict = y_S_predict + peak_amplitude_df.diff_peak_S

    if fitting_type == 'with_site':
        # get the annoying categorical keys
        try:
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
        except:
            raise NameError(f"Index {site_term_column} doesn't exist!")
        if type == 'P':    
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
            if secondary_calibration:
                M_predict = M_predict.array + peak_amplitude_df.diff_peak_P.array/reg.params['magnitude']

        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
            if secondary_calibration:
                M_predict = M_predict.array + peak_amplitude_df.diff_peak_S.array/reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')

    elif fitting_type == 'with_attenuation':
        # get the annoying categorical keys C(region)[ridgecrest]:distance_in_km 
        try:
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
            site_attenu_keys = np.array([f'C(region)[{region}]:distance_in_km' for region in peak_amplitude_df['region']])
        except:
            raise NameError(f"Index {site_term_column} doesn't exist!")

        if type == 'P':    
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km) \
                        - np.array(reg.params[site_attenu_keys]) * np.array(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km) \
                        - np.array(reg.params[site_attenu_keys]) * np.array(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')
        
    elif fitting_type == 'without_site':
        if type == 'P':
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - reg.params['Intercept'] \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - reg.params['Intercept'] \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')

    else:
        raise NameError('Fitting type is undefined!')
        
    return M_predict

def get_mean_magnitude(peak_amplitude_df, M_predict):
    temp_df = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df['predicted_M'] = M_predict
    temp_df = temp_df.groupby(temp_df['event_id']).aggregate(np.nanmedian)

    temp_df2 = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df2['predicted_M_std'] = M_predict
    temp_df2 = temp_df2.groupby(temp_df2['event_id']).aggregate(np.nanstd)
    
    temp_df = pd.concat([temp_df, temp_df2['predicted_M_std']], axis=1)

    return temp_df

def filter_by_channel_number(peak_amplitude_df, min_channel):
    """To remove the measurements from few channels (< min_channel)"""
    event_channel_count = peak_amplitude_df.groupby(['event_id'])['event_id'].count()
    channel_count = event_channel_count.values
    event_id = event_channel_count.index
    event_id = event_id[channel_count >= min_channel]

    return peak_amplitude_df[peak_amplitude_df['event_id'].isin(event_id)]

def split_P_S_dataframe(peak_amplitude_df, site_term_column):
    # use P and S separately to do the regression
    peak_amplitude_df_P = peak_amplitude_df[['event_id', 'channel_id', 'region', 'peak_P', 'magnitude', 'distance_in_km', 'snrP', site_term_column]]
    peak_amplitude_df_S = peak_amplitude_df[['event_id', 'channel_id', 'region', 'peak_S', 'magnitude', 'distance_in_km', 'snrS', site_term_column]]

    # Remove some extreme data outliers before fitting
    peak_amplitude_df_P = peak_amplitude_df_P.dropna()
    peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.peak_P>0]
    peak_amplitude_df_P = peak_amplitude_df_P.drop(peak_amplitude_df_P[(peak_amplitude_df_P.peak_P > 1e3)].index)

    peak_amplitude_df_S = peak_amplitude_df_S.dropna()
    peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.peak_S>0]
    peak_amplitude_df_S = peak_amplitude_df_S.drop(peak_amplitude_df_S[(peak_amplitude_df_S.peak_S > 1e3)].index)

    return peak_amplitude_df_P, peak_amplitude_df_S

def estimate_magnitude(results_output_dir, regression_dir, nearby_channel_number, fitting_type='without_site', site_term_column='region_site', 
                       min_channel=None, magnitude_threshold=None, snr_threshold=None, M_std_percentile=None, secondary_calibration=None):
    """High level function to estimate magnitude"""
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    if magnitude_threshold is not None:
        peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude>=magnitude_threshold[0]) & (peak_amplitude_df.magnitude<=magnitude_threshold[1])]

    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df, site_term_column)
    print(peak_amplitude_df_P.shape)
    if snr_threshold is not None:
        peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.snrP >= snr_threshold]
        peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.snrS >= snr_threshold]

    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    print(peak_amplitude_df_P.shape)

    # load regression with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   

    M_P = calculate_magnitude_from_strain(peak_amplitude_df_P, regP, 'P', fitting_type=fitting_type, 
                                        site_term_column=site_term_column, secondary_calibration=secondary_calibration)
    M_S = calculate_magnitude_from_strain(peak_amplitude_df_S, regS, 'S', fitting_type=fitting_type, 
                                        site_term_column=site_term_column, secondary_calibration=secondary_calibration)

    temp_df_P = get_mean_magnitude(peak_amplitude_df_P, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df_S, M_S)

    if M_std_percentile:
        temp_df_P = temp_df_P[temp_df_P.predicted_M_std < np.percentile(temp_df_P.predicted_M_std.dropna(), M_std_percentile)]
        temp_df_S = temp_df_S[temp_df_S.predicted_M_std < np.percentile(temp_df_S.predicted_M_std.dropna(), M_std_percentile)]

    return temp_df_P, temp_df_S #, M_P, M_S 


def plot_magnitude_seaborn(df_magnitude):
    sns.set_theme(style="ticks", font_scale=2)

    g = sns.JointGrid(data=df_magnitude, x="magnitude", y="predicted_M", marginal_ticks=True,
    xlim=(1, 7), ylim=(1, 7), height=10, space=0.3)


    # Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

    # Add the joint and marginal histogram plots 03012d
    g.plot_joint(
    sns.histplot, discrete=(False, False), bins=(20, 20),
    cmap="dark:#fcd9bb_r", pmax=.7, cbar=True, cbar_ax=cax, cbar_kws={'label':'counts', 'spacing': 'proportional'})
    g.plot_marginals(sns.histplot, element="step", color="#c4a589") # light:#9C4D4D

    # ii_M4 = df_magnitude.magnitude >= 4 
    # g.ax_joint.plot(df_magnitude[ii_M4].magnitude, df_magnitude[ii_M4].predicted_M, 'k.', alpha=1)
    g.ax_joint.plot([1,8], [1,8], 'k-', linewidth = 2)
    g.ax_joint.plot([1,8], [2,9], 'k--', linewidth = 1)
    g.ax_joint.plot([1,8], [0,7], 'k--', linewidth = 1)
    g.ax_joint.set_xlabel('catalog magnitude')
    g.ax_joint.set_ylabel('predicted magnitude')

    return g

# %%
# ========================== work on Combined results from all Ridgecrest + Mammoth S + Mammoth N ================================
# First check how well the regression relation can be used to calculate Magnitude
# load the results from combined regional site terms t
min_channel=100
M_std_percentile = 90
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
regression_dir = f'regression_results_smf_weighted_{min_channel}_channel_at_least' # 'regression_results_smf_M4'
site_term_column = 'region_site'
fitting_type = 'with_site'
nearby_channel_numbers = [100]#[-1, 10, 20, 50, 100]
secondary_calibration = False
output_label = []

# List to hold the estiamted magnitude
temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    temp_df_P, temp_df_S = estimate_magnitude(results_output_dir, regression_dir, nearby_channel_number, fitting_type, 
                            site_term_column, snr_threshold=10, min_channel=min_channel, M_std_percentile=M_std_percentile, 
                            secondary_calibration=secondary_calibration)
    
    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

    gP = plot_magnitude_seaborn(temp_df_P)
    gS = plot_magnitude_seaborn(temp_df_S)    

    if secondary_calibration:
        output_label='_calibrated'

    gP.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_P_{nearby_channel_number}_seaborn{output_label}.png",bbox_inches='tight')
    gP.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_P_{nearby_channel_number}_seaborn{output_label}.pdf",bbox_inches='tight')
    gS.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn{output_label}.png",bbox_inches='tight')
    gS.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn{output_label}.pdf",bbox_inches='tight')
    plt.close('all')


temp_df_P[abs(temp_df_P.predicted_M - temp_df_P.magnitude) > 1.5]

#%% 
# ========================== work on the results from Ridgecrest ================================
# First check how well the regression relation can be used to calculate Magnitude
#% load the results from combined regional site terms t
min_channel=100
M_std_percentile = 90
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
regression_dir = f'regression_results_smf_weighted_{min_channel}_channel_at_least' # 'regression_results_smf_M4'
site_term_column='combined_channel_id'
fitting_type = 'with_site'
nearby_channel_numbers = [-1, 100, 50, 20, 10]
# List to hold the estiamted magnitude
temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    temp_df_P, temp_df_S = estimate_magnitude(results_output_dir, regression_dir, nearby_channel_number, fitting_type, site_term_column, 
                                            snr_threshold=10, min_channel=min_channel, M_std_percentile=M_std_percentile)
    
    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

    gP = plot_magnitude_seaborn(temp_df_P)
    gS = plot_magnitude_seaborn(temp_df_S)
    gP.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_P_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
    gP.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_P_{nearby_channel_number}_seaborn.pdf",bbox_inches='tight')
    gS.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
    gS.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.pdf",bbox_inches='tight')
    plt.close('all')


#%% 
# ========================== work on the results from Mammoth South ================================
# First check how well the regression relation can be used to calculate Magnitude
#% load the results from combined regional site terms t
min_channel=100
M_std_percentile = 90
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
regression_dir = f'regression_results_smf_weighted_{min_channel}_channel_at_least'
site_term_column='combined_channel_id'
nearby_channel_numbers = [-1, 100, 50, 20, 10]
fitting_type = 'with_site'
magnitude_threshold = [2, 10]

# List to hold the estiamted magnitude
temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    temp_df_P, temp_df_S = estimate_magnitude(results_output_dir, regression_dir, nearby_channel_number, fitting_type, site_term_column, 
                                            magnitude_threshold=magnitude_threshold, snr_threshold=10, min_channel=min_channel, M_std_percentile=M_std_percentile)
    
    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

    gP = plot_magnitude_seaborn(temp_df_P)
    gS = plot_magnitude_seaborn(temp_df_S)
    gP.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_P_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
    gP.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_P_{nearby_channel_number}_seaborn.pdf",bbox_inches='tight')
    gS.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
    gS.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.pdf",bbox_inches='tight')
    plt.close('all')


#%% 
# ========================== work on the results from Mammoth North ================================
# First check how well the regression relation can be used to calculate Magnitude
#% load the results from combined regional site terms t
min_channel=100
M_std_percentile = 90
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
regression_dir = f'regression_results_smf_weighted_{min_channel}_channel_at_least'
site_term_column='combined_channel_id'
fitting_type = 'with_site'
nearby_channel_numbers = [-1, 100, 50, 20, 10]
magnitude_threshold = [2, 10]

# List to hold the estiamted magnitude
temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    temp_df_P, temp_df_S = estimate_magnitude(results_output_dir, regression_dir, nearby_channel_number, fitting_type, site_term_column, 
                                            magnitude_threshold=magnitude_threshold, snr_threshold=10, min_channel=min_channel, M_std_percentile=M_std_percentile)
    
    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

    gP = plot_magnitude_seaborn(temp_df_P)
    gS = plot_magnitude_seaborn(temp_df_S)
    gP.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_P_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
    gP.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_P_{nearby_channel_number}_seaborn.pdf",bbox_inches='tight')
    gS.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
    gS.savefig(results_output_dir + '/' + regression_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.pdf",bbox_inches='tight')
    plt.close('all')

# %%
# %%
# # ========================== work on Combined results from Mammoth S + Mammoth N ================================
# # First check how well the regression relation can be used to calculate Magnitude
# # load the results from combined regional site terms t
# results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_MM'
# regression_dir = 'regression_results_smf' # 'regression_results_smf_M4'
# site_term_column = 'region_site'
# fitting_type = 'with_site'
# nearby_channel_numbers = [-1, 10, 20, 50, 100]

# # List to hold the estiamted magnitude
# temp_df_P_list = []
# temp_df_S_list = []

# for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
#     temp_df_P, temp_df_S = estimate_magnitude(results_output_dir, regression_dir, nearby_channel_number, fitting_type, site_term_column)
#     temp_df_P_list.append(temp_df_P)
#     temp_df_S_list.append(temp_df_S)

# plot_magnitude_prediction(temp_df_P_list, temp_df_S_list, label_extrapolate=False)
# plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")

# plot_magnitude_prediction_residual(temp_df_P_list, temp_df_S_list, label_extrapolate=False)
# plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude_residual.png")

# for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
#     figP, figS = show_all_results(results_output_dir, regression_dir, site_term_column, fitting_type, nearby_channel_number)
#     plt.figure(figP.number)
#     plt.savefig(results_output_dir + '/' + regression_dir + f"/P_predicted_strain_and_magnitude_{nearby_channel_number}.png")

#     plt.figure(figS.number)
#     plt.savefig(results_output_dir + '/' + regression_dir + f"/S_predicted_strain_and_magnitude_{nearby_channel_number}.png")
#     plt.close('all')


# %% ==================== WILL BE REMOVED SOON! =========================================
# def plot_magnitude_prediction(temp_df_P_list, temp_df_S_list, label_extrapolate=True):
#     horizontal_shift = [0, 0, 0, 0, 0] #[-0.015, -0.005, 0.005, 0.015]
#     cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']
#     fig, ax = plt.subplots(2, 1, figsize=(10, 20), sharex=True, sharey=True)
#     for ii in range(len(temp_df_P_list)):
#         temp_df_P = temp_df_P_list[ii]
#         temp_df_S = temp_df_S_list[ii]

#         ax[0].errorbar(temp_df_P.magnitude + horizontal_shift[ii], temp_df_P.predicted_M, yerr=temp_df_P.predicted_M_std, marker='o', linestyle='none')
#         ax[0].plot([0, 10], [0, 10], '-k', zorder=1)
#         ax[0].set_xlim(2, 8.5)
#         ax[0].set_ylim(2, 8.5)
#         ax[0].set_ylabel('P predicted M')
#         ax[0].set_xlabel('true M')
#         ax[0].xaxis.set_tick_params(which='both',labelbottom=True)

#         ax[1].errorbar(temp_df_S.magnitude + horizontal_shift[ii], temp_df_S.predicted_M, yerr=temp_df_S.predicted_M_std, marker='o', linestyle='none')
#         ax[1].plot([0, 10], [0, 10], '-k', zorder=1)
#         ax[1].set_xlim(0, 8.5)
#         ax[1].set_ylim(0, 8.5)
#         ax[1].set_ylabel('S predicted M')
#         ax[1].set_xlabel('true M')

#     if label_extrapolate:
#         ax[0].vlines(x=4, ymin=0, ymax=10, linestyle='--', color='k')
#         ax[0].text(2.5, 5.75, 'regression')
#         ax[0].text(4.5, 5.75, 'prediction')
#         ax[1].vlines(x=4, ymin=0, ymax=10, linestyle='--', color='k')
#         ax[1].text(2.5, 5.75, 'regression')
#         ax[1].text(4.5, 5.75, 'prediction')

# def plot_magnitude_prediction_residual(temp_df_P_list, temp_df_S_list, label_extrapolate=True):
#     horizontal_shift = [0, 0, 0, 0, 0] #[-0.015, -0.005, 0.005, 0.015]
#     cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']
#     fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
#     for ii in range(len(temp_df_P_list)):
#         temp_df_P = temp_df_P_list[ii]
#         temp_df_S = temp_df_S_list[ii]

#         ax[0].errorbar(temp_df_P.magnitude + horizontal_shift[ii], temp_df_P.predicted_M - temp_df_P.magnitude, yerr=temp_df_P.predicted_M_std, marker='o', linestyle='none')
#         ax[0].plot([0, 10], [0, 0], '-k', zorder=1)
        
#         ax[0].set_xlim(0, 8.5)
#         ax[0].set_ylim(-4, 4)
#         ax[0].set_ylabel('P predicted M residual')
#         ax[0].set_xlabel('true M')
#         ax[0].xaxis.set_tick_params(which='both',labelbottom=True)

#         ax[1].errorbar(temp_df_S.magnitude + horizontal_shift[ii], temp_df_S.predicted_M - temp_df_S.magnitude, yerr=temp_df_S.predicted_M_std, marker='o', linestyle='none')
#         ax[1].plot([0, 10], [0, 0], '-k', zorder=1)
#         ax[1].set_xlim(0, 8.5)
#         ax[1].set_ylim(-4, 4)
#         ax[1].set_ylabel('S predicted M residual')
#         ax[1].set_xlabel('true M')

#     if label_extrapolate:
#         ax[0].vlines(x=4, ymin=-10, ymax=10, linestyle='--', color='k')
#         ax[0].text(2.5, 2.5, 'regression')
#         ax[0].text(4.5, 2.5, 'prediction')
#         ax[1].vlines(x=4, ymin=-10, ymax=10, linestyle='--', color='k')
#         ax[1].text(2.5, 2.5, 'regression')
#         ax[1].text(4.5, 2.5, 'prediction')

# def show_all_results(results_output_dir, regression_dir, site_term_column, fitting_type, nearby_channel_number):
#     # magnitude estimation
#     temp_df_P, temp_df_S = estimate_magnitude(results_output_dir, regression_dir, nearby_channel_number, fitting_type, site_term_column)

#     # Get the predicted strain rate
#     peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')

#     # load regression with different regional site terms
#     regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
#     peak_P_predicted = regP.predict(peak_amplitude_df)
#     # load regression with different regional site terms
#     regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
#     peak_S_predicted = regS.predict(peak_amplitude_df)

#     figP = plot_fitting_and_magnitude(temp_df_P, 'peak_P', peak_amplitude_df, peak_P_predicted)
#     figS = plot_fitting_and_magnitude(temp_df_S, 'peak_S', peak_amplitude_df, peak_S_predicted)

#     return figP, figS

# def plot_fitting_and_magnitude(temp_df, phase_key, peak_amplitude_df, peak_predicted):
#     # Plot figure
#     fig, ax = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'height_ratios':[3,1]})

#     data_lim = (-2, 4)
#     gca = ax[0, 0]
#     gca.plot([0, 10], [0, 10], '-k', zorder=1)
#     # gca.scatter(np.log10(peak_amplitude_df['peak_P']), peak_P_predicted, s=10, c=peak_amplitude_df.event_label, marker='o', alpha=0.1, cmap='jet')
#     gca.plot(np.log10(peak_amplitude_df[phase_key]), peak_predicted, '.', alpha=0.1)
#     gca.set_ylabel('predicted log10(E)')
#     gca.set_xlabel('measured log10(E)')

#     gca.set_xticks(np.arange(-2, 10))
#     gca.set_yticks(np.arange(-2, 10))

#     gca.set_xlim(data_lim)
#     gca.set_ylim(data_lim)
#     gca.grid()
#     gca.annotate('(a)', xy=(-0.1, 1.05), xycoords=gca.transAxes)


#     data_lim = (-2, 4)
#     gca = ax[1, 0]
#     gca.plot([0, 10], [0, 0], '-k', zorder=1)
#     # gca.scatter(np.log10(peak_amplitude_df['peak_P']), np.log10(peak_amplitude_df['peak_P']) - peak_P_predicted, s=10, c=peak_amplitude_df.event_label, marker='o', alpha=0.1, cmap='jet')
#     gca.plot(np.log10(peak_amplitude_df[phase_key]), np.log10(peak_amplitude_df[phase_key]) - peak_predicted, '.', alpha=0.1)
#     gca.set_ylabel('log10(E) error')
#     gca.set_xlabel('measured log10(E)')

#     gca.set_xlim(data_lim)
#     gca.set_ylim(-2, 2)
#     gca.grid()
#     gca.annotate('(b)', xy=(-0.1, 1.05), xycoords=gca.transAxes)


#     data_lim = (-2, 7)
#     gca = ax[0, 1]
#     gca.plot([0, 10], [0, 10], '-k', zorder=1)
#     gca.errorbar(temp_df.magnitude, temp_df.predicted_M, yerr=temp_df.predicted_M_std, marker='o', linestyle='none')
#     gca.set_ylabel('predicted M')
#     gca.set_xlabel('catalog M')
#     # gca.vlines(x=4, ymin=-10, ymax=10, linestyle='--', color='k')
#     # gca.text(2.5, 6.5, 'regression')
#     # gca.text(4.5, 6.5, 'prediction')

#     gca.set_xticks(np.arange(0, 10))
#     gca.set_yticks(np.arange(0, 10))

#     gca.set_xlim(data_lim)
#     gca.set_ylim(data_lim)
#     gca.grid()
#     gca.annotate('(c)', xy=(-0.1, 1.05), xycoords=gca.transAxes)


#     data_lim = (-2, 7)
#     gca = ax[1, 1]
#     gca.plot([0, 10], [0, 0], '-k', zorder=1)
#     gca.errorbar(temp_df.magnitude, temp_df.magnitude - temp_df.predicted_M, yerr=temp_df.predicted_M_std, marker='o', linestyle='none')
#     gca.set_ylabel('M error')
#     gca.set_xlabel('catalog M')
#     # gca.vlines(x=4, ymin=-10, ymax=10, linestyle='--', color='k')
#     # gca.text(2.5, 1.5, 'regression')
#     # gca.text(4.5, 1.5, 'prediction')

#     gca.set_xticks(np.arange(0, 10))

#     gca.set_xlim(data_lim)
#     gca.set_ylim(-2, 2)
#     gca.grid()
#     gca.annotate('(d)', xy=(-0.1, 1.05), xycoords=gca.transAxes)
#     return fig