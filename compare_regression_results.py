#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# import the plotting functions
from plotting_functions import *

# ==============================  Ridgecrest data ========================================
#%% Specify the file names
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
#results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain'
das_pick_file_name = '/peak_amplitude_M3+.csv'

# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/home/yinjx/kuafu/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain_rate'
results_output_dir = '/home/yinjx/kuafu/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain'
das_pick_file_name = '/peak_amplitude_M3+.csv'

# ==============================  Mammoth data - South========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_name = '/Mammoth_South_Scaling_M3.csv'

# ==============================  Mammoth data - North========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_name = '/Mammoth_North_Scaling_M3.csv'

#%% specify where the regression results are
regression_results_dir = results_output_dir + '/regression_results_smf'

# make a directory to store the regression results in text
regression_text = regression_results_dir + '/regression_results_txt'
if not os.path.exists(regression_text):
    os.mkdir(regression_text)


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
    temp1= np.arange(0, DAS_index.max()+1) # original channel number
    temp2 = temp1 // nearby_channel_number # combined channel number
    peak_amplitude_df['combined_channel_id'] = temp2[np.array(peak_amplitude_df.channel_id).astype('int')]
    return peak_amplitude_df

def store_model_parameters_to_df(reg, parameter_df, combined_channel_number, digits=3):
    parameter_df['combined_channels'][i_model] = combined_channel_number
    parameter_df['magnitude'][i_model] = round(reg.params[-2],digits)
    parameter_df['distance'][i_model] = round(reg.params[-1],digits)
    parameter_df['magnitude_err'][i_model] = round(np.sqrt(reg.cov_params().iloc[-2][-2]),digits)
    parameter_df['distance_err'][i_model] = round(np.sqrt(reg.cov_params().iloc[-1][-1]),digits)
    return parameter_df

# %% Compare the regression parameters and site terms
fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model

# DataFrame to store parameters for all models
P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'], 
index = np.arange(len(combined_channel_number_list)))
S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
index = np.arange(len(combined_channel_number_list)))

for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    if combined_channel_number == 1:
        regP = sm.load(regression_results_dir + "/P_regression_all_events_with_site_terms.pickle")
        regS = sm.load(regression_results_dir + "/S_regression_all_events_with_site_terms.pickle")

    elif combined_channel_number == -1:
        regP = sm.load(regression_results_dir + "/P_regression_all_events_no_site_terms.pickle")
        regS = sm.load(regression_results_dir + "/S_regression_all_events_no_site_terms.pickle")
    else:
        regP = sm.load(regression_results_dir + f"/P_regression_all_events_with_combined_site_terms_{combined_channel_number}chan.pickle")
        regS = sm.load(regression_results_dir + f"/S_regression_all_events_with_combined_site_terms_{combined_channel_number}chan.pickle")

        peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)

    # output to text files
    with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{combined_channel_number}chan.txt", "w") as text_file:
        text_file.write(regP.summary().as_text())
    with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{combined_channel_number}chan.txt", "w") as text_file:
        text_file.write(regS.summary().as_text())

    # Store the parameters 
    P_parameters_comparison = store_model_parameters_to_df(regP, P_parameters_comparison, combined_channel_number)
    S_parameters_comparison = store_model_parameters_to_df(regS, S_parameters_comparison, combined_channel_number)
    
# Compare all the site terms
    site_term_P = regP.params[:-2]
    site_term_S = regS.params[:-2]

    if combined_channel_number == 1:
        ax[0].plot(peak_amplitude_df['channel_id'].unique(), site_term_P, label=f'Individual site terms, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(peak_amplitude_df['channel_id'].unique(), site_term_S, label=f'Individual site terms, Cond.# {regS.condition_number:.2f}')
    elif combined_channel_number == -1:
        ax[0].hlines(site_term_P, xmin=DAS_index.min(), xmax=DAS_index.max(), color='k', label=f'Same site terms, Cond.# {regP.condition_number:.2f}')
        ax[1].hlines(site_term_S, xmin=DAS_index.min(), xmax=DAS_index.max(), color='k', label=f'Same site terms, Cond.# {regS.condition_number:.2f}')
    else:
        ax[0].plot(peak_amplitude_df['combined_channel_id'].unique() * combined_channel_number, site_term_P, label=f'{combined_channel_number} channels, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(peak_amplitude_df['combined_channel_id'].unique() * combined_channel_number, site_term_S, label=f'{combined_channel_number} channels, Cond.# {regS.condition_number:.2f}')

    # reset the regression models
    del regP, regS

ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
ax[0].set_xlabel('Channel number')
ax[1].set_xlabel('Channel number')
ax[0].set_ylabel('Site terms P')
ax[1].set_ylabel('Site terms S')
#ax[1].invert_xaxis()

plt.savefig(regression_results_dir + '/compare_site_terms.png', bbox_inches='tight')

P_parameters_comparison.to_csv(regression_results_dir + '/parameter_comparison_P.txt', index=False, sep='\t')
S_parameters_comparison.to_csv(regression_results_dir + '/parameter_comparison_S.txt', index=False, sep='\t')
# %%

# %%
