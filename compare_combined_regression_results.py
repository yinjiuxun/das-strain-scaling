#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# import the plotting functions
from plotting_functions import *

# ==============================  Ridgecrest + Olancha data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/combined_strain_scaling'
#results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain'
das_pick_file_name = '/peak_amplitude_region_site_all.csv'

#%% specify where the regression results are
regression_results_dir = results_output_dir + '/regression_results_smf_M3'

# make a directory to store the regression results in text
regression_text = regression_results_dir + '/regression_results_txt'
if not os.path.exists(regression_text):
    os.mkdir(regression_text)

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
        regP = sm.load(regression_results_dir + f"/P_regression_region_site_terms_{combined_channel_number}chan.pickle")
        regS = sm.load(regression_results_dir + f"/S_regression_region_site_terms_{combined_channel_number}chan.pickle")

    # output to text files
    with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{combined_channel_number}chan.txt", "w") as text_file:
        text_file.write(regP.summary().as_text())
    with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{combined_channel_number}chan.txt", "w") as text_file:
        text_file.write(regS.summary().as_text())

    # Store the parameters 
    P_parameters_comparison = store_model_parameters_to_df(regP, P_parameters_comparison, combined_channel_number)
    S_parameters_comparison = store_model_parameters_to_df(regS, S_parameters_comparison, combined_channel_number)

    # reset the regression models
    del regP, regS

P_parameters_comparison.to_csv(regression_results_dir + '/parameter_comparison_P.txt', index=False, sep='\t')
S_parameters_comparison.to_csv(regression_results_dir + '/parameter_comparison_S.txt', index=False, sep='\t')
# %%

# %%
