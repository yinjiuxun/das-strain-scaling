#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

# import the plotting functions
from plotting_functions import *
from utility_functions import *

# ==============================  Ridgecrest data ========================================
#%% Specify the file names
# results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
# das_pick_file_name = '/peak_amplitude_M3+.csv'
# region_label = 'ridgecrest'

results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events'
das_pick_file_name = '/peak_amplitude.csv'
region_label = 'ridgecrest'
channel_space = 8/1e3
# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling'
das_pick_file_name = '/peak_amplitude_M3+.csv'
region_label = 'olancha'

# ==============================  Mammoth data - South========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events'
das_pick_file_name = '/peak_amplitude.csv'
region_label = 'mammothS'
channel_space = 10/1e3

# ==============================  Mammoth data - North========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events'
das_pick_file_name = '/peak_amplitude.csv'
region_label = 'mammothN'
channel_space = 10/1e3

#%% load the peak amplitude results
# Load the peak amplitude results
snr_threshold = 10
magnitude_threshold = [2, 10]
peak_amplitude_df = pd.read_csv(das_pick_file_folder + '/' + das_pick_file_name)

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf_weighted'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.snrP >=snr_threshold) | (peak_amplitude_df.snrS >=snr_threshold)]
peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >=magnitude_threshold[0]) & (peak_amplitude_df.magnitude <=magnitude_threshold[1])]

# peak_amplitude_df = peak_amplitude_df.dropna()
DAS_index = peak_amplitude_df.channel_id.unique().astype('int')

#%% Functions used here
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
def plot_prediction_vs_measure_seaborn(peak_comparison_df, xy_range, phase):
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
    g.plot_joint(sns.histplot, discrete=(False, False), cmap="light:#4D4D9C", pmax=.2, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", color="#4D4D9C")

    g.ax_joint.plot(xy_range, xy_range, 'k-', linewidth = 2)
    g.ax_joint.set_xlabel('measured peak')
    g.ax_joint.set_ylabel('calculated peak')
    return g

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
    
# Compare all the site terms
    site_term_P = regP.params[:-2]
    site_term_S = regS.params[:-2]

    if combined_channel_number == 1:
        ax[0].plot(combined_channel_id*channel_space, site_term_P, label=f'Individual site terms, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(combined_channel_id*channel_space, site_term_S, label=f'Individual site terms, Cond.# {regS.condition_number:.2f}')
    elif combined_channel_number == -1:
        ax[0].hlines(site_term_P, xmin=DAS_index.min()*channel_space, xmax=DAS_index.max()*channel_space, color='k', label=f'Same site terms')#, Cond.# {regP.condition_number:.2f}')
        ax[1].hlines(site_term_S, xmin=DAS_index.min()*channel_space, xmax=DAS_index.max()*channel_space, color='k', label=f'Same site terms')  #Cond.# {regS.condition_number:.2f}')
    else:
        ax[0].plot(combined_channel_id * combined_channel_number * channel_space, np.array(site_term_P), '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(combined_channel_id * combined_channel_number * channel_space, site_term_S, '-', label=f'{combined_channel_number} channels')#, Cond.# {regS.condition_number:.2f}')

    # reset the regression models
    #del regP, regS

ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
ax[0].set_xlabel('Distance along cable (km)')
ax[1].set_xlabel('Distance along cable (km)')
ax[0].set_ylabel('Site terms P')
ax[1].set_ylabel('Site terms S')
#ax[1].invert_xaxis()

plt.figure(fig.number)
plt.savefig(regression_results_dir + '/compare_site_terms.png', bbox_inches='tight')

# %%
# ==============================  Looking into the multiple array case ========================================
# Specify the file names
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf_weighted'

# %% Compare the regression parameters and site terms
fig, ax = plt.subplots(2, 1, figsize=(10, 18), sharex=True, sharey=True) # north
fig2, ax2 = plt.subplots(2, 1, figsize=(10, 18), sharex=True, sharey=True) # south
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{combined_channel_number}.csv')

    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")

    # peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)
    # combined_channel_id = np.sort(peak_amplitude_df.combined_channel_id.unique())
    # peak_amplitude_df = add_event_label(peak_amplitude_df)

    
# extract the site terms for different arrays
    combined_channel_north, site_term_P_north = process_region_param_keys(regP, 'mammothN')
    combined_channel_north, site_term_S_north = process_region_param_keys(regS, 'mammothN')

    combined_channel_south, site_term_P_south = process_region_param_keys(regP, 'mammothS')
    combined_channel_south, site_term_S_south = process_region_param_keys(regS, 'mammothS')
   
    if combined_channel_number == -1:
        ax[0].hlines(site_term_P_north, xmin=0, xmax=5000, color='k', label=f'Same site terms')#, Cond.# {regP.condition_number:.2f}')
        ax[1].hlines(site_term_S_north, xmin=0, xmax=5000, color='k', label=f'Same site terms')#, Cond.# {regS.condition_number:.2f}')
                
        ax2[0].hlines(site_term_P_south, xmin=0, xmax=5000, color='k', label=f'Same site terms')#, Cond.# {regP.condition_number:.2f}')
        ax2[1].hlines(site_term_S_south, xmin=0, xmax=5000, color='k', label=f'Same site terms')#, Cond.# {regS.condition_number:.2f}')
    else:
        ax[0].plot(combined_channel_north * combined_channel_number, site_term_P_north, '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(combined_channel_north * combined_channel_number, site_term_S_north, '-', label=f'{combined_channel_number} channels')#, Cond.# {regS.condition_number:.2f}')

        ax2[0].plot(combined_channel_south * combined_channel_number, site_term_P_south, '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
        ax2[1].plot(combined_channel_south * combined_channel_number, site_term_S_south, '-', label=f'{combined_channel_number} channels')#, Cond.# {regS.condition_number:.2f}')
    # reset the regression models
    #del regP, regS

ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
ax[0].set_xlabel('Channel number')
ax[1].set_xlabel('Channel number')
ax[0].set_ylabel('Site terms P')
ax[1].set_ylabel('Site terms S')
ax[0].set_title('Mammoth North array')

ax2[0].legend(fontsize=12)
ax2[1].legend(fontsize=12)
ax2[0].xaxis.set_tick_params(which='both',labelbottom=True)
ax2[0].set_xlabel('Channel number')
ax2[1].set_xlabel('Channel number') 
ax2[0].set_ylabel('Site terms P')
ax2[1].set_ylabel('Site terms S')
ax2[0].set_title('Mammoth South array')
#ax[1].invert_xaxis()

plt.figure(fig.number)
plt.savefig(regression_results_dir + '/compare_site_terms_north.png', bbox_inches='tight')

plt.figure(fig2.number)
plt.savefig(regression_results_dir + '/compare_site_terms_south.png', bbox_inches='tight')
# %%
# #Test parameters from strain meter (Not right)
# fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
# # combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
# combined_channel_number_list = [100]
# # DataFrame to store parameters for all models
# P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'], 
# index = np.arange(len(combined_channel_number_list)))
# S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
# index = np.arange(len(combined_channel_number_list)))

# for i_model, combined_channel_number in enumerate(combined_channel_number_list):

#     regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
#     regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
#     # apply parameters from strain meter
#     regP.params['magnitude'] = 0.92
#     regP.params['np.log10(distance_in_km)'] = -1.45

#     regS.params['magnitude'] = 0.92
#     regS.params['np.log10(distance_in_km)'] = -1.45

#     peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)
#     combined_channel_id = np.sort(peak_amplitude_df.combined_channel_id.unique())
#     peak_amplitude_df = add_event_label(peak_amplitude_df)

#     y_P_predict = regP.predict(peak_amplitude_df)
#     y_S_predict = regS.predict(peak_amplitude_df)

#     temp_peaks = np.array([np.array(peak_amplitude_df.peak_P),
#               np.array(peak_amplitude_df.peak_S), 
#               np.array(10**y_P_predict), 
#               np.array(10**y_S_predict)]).T
#     peak_comparison_df = pd.DataFrame(data=temp_peaks,
#                                   columns=['peak_P', 'peak_S', 'peak_P_predict', 'peak_S_predict'])
    
#     g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 1000], phase='P')
#     g.savefig(regression_results_dir + f'/P_validate_predicted_combined_site_terms_{combined_channel_number}chan_strainmeter_seaborn.png')

#     g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 1000], phase='S')
#     g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_strainmeter_seaborn.png')
#     # plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict, y_S_predict, (-2.0, 2), 
#     # regression_results_dir + f'/validate_predicted__combined_site_terms_{combined_channel_number}chan.png')