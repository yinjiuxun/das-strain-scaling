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

# ==============================  Ridgecrest data ========================================
#%% Specify the file names
# results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
# das_pick_file_name = '/peak_amplitude_M3+.csv'
# region_label = 'ridgecrest'

results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate_snr'
das_pick_file_name = '/peak_amplitude_M3+.csv'
region_label = 'ridgecrest'

# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling'
das_pick_file_name = '/peak_amplitude_M3+.csv'
region_label = 'olancha'

# ==============================  Mammoth data - South========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_name = '/Mammoth_South_Scaling_M3.csv'
region_label = 'mammothS'

# ==============================  Mammoth data - North========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_name = '/Mammoth_North_Scaling_M3.csv'
region_label = 'mammothN'

#%% load the peak amplitude results
# Load the peak amplitude results
peak_amplitude_df = pd.read_csv(results_output_dir + '/' + das_pick_file_name)

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)


peak_amplitude_df = peak_amplitude_df.dropna()
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
    
    plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict, y_S_predict, (1.0, 5.5), 
    regression_results_dir + f'/validate_predicted__combined_site_terms_{combined_channel_number}chan.png')
    
# Compare all the site terms
    site_term_P = regP.params[:-2]
    site_term_S = regS.params[:-2]

    if combined_channel_number == 1:
        ax[0].plot(combined_channel_id, site_term_P, label=f'Individual site terms, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(combined_channel_id, site_term_S, label=f'Individual site terms, Cond.# {regS.condition_number:.2f}')
    elif combined_channel_number == -1:
        ax[0].hlines(site_term_P, xmin=DAS_index.min(), xmax=DAS_index.max(), color='k', label=f'Same site terms, Cond.# {regP.condition_number:.2f}')
        ax[1].hlines(site_term_S, xmin=DAS_index.min(), xmax=DAS_index.max(), color='k', label=f'Same site terms, Cond.# {regS.condition_number:.2f}')
    else:
        ax[0].plot(combined_channel_id * combined_channel_number, np.array(site_term_P), '-', label=f'{combined_channel_number} channels, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(combined_channel_id * combined_channel_number, site_term_S, '-', label=f'{combined_channel_number} channels, Cond.# {regS.condition_number:.2f}')

    # reset the regression models
    #del regP, regS

ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
ax[0].set_xlabel('Channel number')
ax[1].set_xlabel('Channel number')
ax[0].set_ylabel('Site terms P')
ax[1].set_ylabel('Site terms S')
#ax[1].invert_xaxis()

plt.figure(fig.number)
plt.savefig(regression_results_dir + '/compare_site_terms.png', bbox_inches='tight')









# %%
# ==============================  Looking into the multiple array case ========================================
# Specify the file names
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_MM'

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf'

# %% Compare the regression parameters and site terms
fig, ax = plt.subplots(2, 1, figsize=(10, 18), sharex=True, sharey=True) # north
fig2, ax2 = plt.subplots(2, 1, figsize=(10, 18), sharex=True, sharey=True) # south
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{combined_channel_number}.csv')

    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")

    # y_P_predict = regP.predict(peak_amplitude_df)
    # y_S_predict = regS.predict(peak_amplitude_df)
    
    # plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict, y_S_predict, (1.0, 6.5), 
    # regression_results_dir + f'/validate_predicted__combined_site_terms_{combined_channel_number}chan.png')
    
# extract the site terms for different arrays
    combined_channel_north, site_term_P_north = process_region_param_keys(regP, 'mammothN')
    combined_channel_north, site_term_S_north = process_region_param_keys(regS, 'mammothN')

    combined_channel_south, site_term_P_south = process_region_param_keys(regP, 'mammothS')
    combined_channel_south, site_term_S_south = process_region_param_keys(regS, 'mammothS')
   
    if combined_channel_number == -1:
        ax[0].hlines(site_term_P_north, xmin=0, xmax=5000, color='k', label=f'Same site terms, Cond.# {regP.condition_number:.2f}')
        ax[1].hlines(site_term_S_north, xmin=0, xmax=5000, color='k', label=f'Same site terms, Cond.# {regS.condition_number:.2f}')
                
        ax2[0].hlines(site_term_P_south, xmin=0, xmax=5000, color='k', label=f'Same site terms, Cond.# {regP.condition_number:.2f}')
        ax2[1].hlines(site_term_S_south, xmin=0, xmax=5000, color='k', label=f'Same site terms, Cond.# {regS.condition_number:.2f}')
    else:
        ax[0].plot(combined_channel_north * combined_channel_number, site_term_P_north, '-', label=f'{combined_channel_number} channels, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(combined_channel_north * combined_channel_number, site_term_S_north, '-', label=f'{combined_channel_number} channels, Cond.# {regS.condition_number:.2f}')

        ax2[0].plot(combined_channel_south * combined_channel_number, site_term_P_south, '-', label=f'{combined_channel_number} channels, Cond.# {regP.condition_number:.2f}')
        ax2[1].plot(combined_channel_south * combined_channel_number, site_term_S_south, '-', label=f'{combined_channel_number} channels, Cond.# {regS.condition_number:.2f}')
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
