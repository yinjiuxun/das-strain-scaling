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

#%% Functions used here
def model_parameters_df(reg, combined_channel_number, digits=3):
    magnitude = round(reg.params[-2],digits)
    distance = round(reg.params[-1],digits)
    magnitude_err = round(np.sqrt(reg.cov_params().iloc[-2][-2]),digits)
    distance_err = round(np.sqrt(reg.cov_params().iloc[-1][-1]),digits)
    parameter_df = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
    data = [[combined_channel_number, magnitude, distance, magnitude_err, distance_err]])
    return parameter_df

def fit_regression(combined_channel_number_list, M_threshold, results_output_dir, regression_results_dir, regression_text):
    # DataFrame to store parameters for all models
    P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err']) 
    S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'])

    for nearby_channel_number in combined_channel_number_list:
        if nearby_channel_number == -1:
            peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_region_site_all.csv')
            peak_amplitude_df_M = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
            regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df_M).fit()
            regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df_M).fit()
        else:
            peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
            peak_amplitude_df_M = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
        # %% Now can fit the data with different regional site terms
            regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_M).fit()
            regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_M).fit()

        regP.save(regression_results_dir + f"/P_regression_region_site_terms_{nearby_channel_number}chan.pickle")
        regS.save(regression_results_dir + f"/S_regression_region_site_terms_{nearby_channel_number}chan.pickle")

    # output to text files
        with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
            text_file.write(regP.summary().as_text())
        with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
            text_file.write(regS.summary().as_text())

    # Store the parameters 
        P_parameters_comparison = pd.concat([P_parameters_comparison, model_parameters_df(regP, nearby_channel_number)], axis=0)
        S_parameters_comparison = pd.concat([S_parameters_comparison, model_parameters_df(regS, nearby_channel_number)], axis=0)

    P_parameters_comparison.to_csv(regression_text + '/parameter_comparison_P.txt', index=False, sep='\t')
    S_parameters_comparison.to_csv(regression_text + '/parameter_comparison_S.txt', index=False, sep='\t')

# ==============================  Ridgecrest data ========================================
#%% Specify the file names
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_name = '/peak_amplitude_M3+.csv'

peak_amplitude_df_ridgecrest = pd.read_csv(results_output_dir + '/' + das_pick_file_name)
peak_amplitude_df_ridgecrest['region'] = 'ridgecrest' # add the region label
DAS_index_ridgecrest = peak_amplitude_df_ridgecrest.channel_id.unique().astype('int')
peak_amplitude_df_ridgecrest = peak_amplitude_df_ridgecrest.dropna()

# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/home/yinjx/kuafu/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_name = '/peak_amplitude_M3+.csv'

peak_amplitude_df_olancha = pd.read_csv(results_output_dir + '/' + das_pick_file_name)
peak_amplitude_df_olancha['region'] = 'olancha' # add the region label
DAS_index_olancha = peak_amplitude_df_olancha.channel_id.unique().astype('int')
peak_amplitude_df_olancha = peak_amplitude_df_olancha.dropna()


#%% Combine the peak results from different regions
results_output_dir = '/kuafu/yinjx/combined_strain_scaling'

#%% Preprocess the data file: combining different channels etc.
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
for nearby_channel_number in combined_channel_number_list:
    if nearby_channel_number == -1:
        peak_amplitude_df = pd.concat((peak_amplitude_df_ridgecrest, peak_amplitude_df_olancha), axis=0)
        peak_amplitude_df = add_event_label(peak_amplitude_df)

        peak_amplitude_df.to_csv(results_output_dir + '/peak_amplitude_region_site_all.csv', index=False)
    else:
        peak_amplitude_df_ridgecrest = combined_channels(DAS_index_ridgecrest, peak_amplitude_df_ridgecrest, nearby_channel_number)
        peak_amplitude_df_olancha = combined_channels(DAS_index_olancha, peak_amplitude_df_olancha, nearby_channel_number)

        peak_amplitude_df = pd.concat((peak_amplitude_df_ridgecrest, peak_amplitude_df_olancha), axis=0)
        # %% Aggregate the columns of region and combined_channel_id to form the regional site terms
        peak_amplitude_df['combined_channel_id']= peak_amplitude_df['combined_channel_id'].astype('str')
        peak_amplitude_df['region_site'] = peak_amplitude_df[['region', 'combined_channel_id']].agg('-'.join, axis=1)

        # Store the processed DataFrame
        peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)

#%% Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

# make a directory to store the regression results in text
regression_text = regression_results_dir + '/regression_results_txt'
if not os.path.exists(regression_text):
    os.mkdir(regression_text)

combined_channel_number_list = [10, 20, 50, 100, -1]
M_threshold = [0, 9]
fit_regression(combined_channel_number_list, M_threshold, results_output_dir, regression_results_dir, regression_text)


# ======================= Below are the part to use the small events to do the regression ===========================
# %% Now only use the smaller earthquakes to do the regression
# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf_M4'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

# make a directory to store the regression results in text
regression_text = regression_results_dir + '/regression_results_txt'
if not os.path.exists(regression_text):
    os.mkdir(regression_text)

combined_channel_number_list = [10, 20, 50, 100, -1]
M_threshold = [0, 4]
fit_regression(combined_channel_number_list, M_threshold, results_output_dir, regression_results_dir, regression_text)

