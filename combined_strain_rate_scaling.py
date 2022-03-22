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


#%% Combine the peak results from different regions
results_output_dir = '/kuafu/yinjx/combined_strain_scaling'

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

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

# concatenate the DataFrame
# combine different regions
peak_amplitude_df = pd.concat((peak_amplitude_df_ridgecrest, peak_amplitude_df_olancha), axis=0)
peak_amplitude_df = add_event_label(peak_amplitude_df)

peak_amplitude_df.to_csv(results_output_dir + '/peak_amplitude_region_site_all.csv', index=False)

# %% Regression 1. Linear regression on the data point (this regression ignores the different site responses)
regP_1 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df).fit()
regS_1 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df).fit()

regP_1.save(regression_results_dir + "/P_regression_all_events_no_site_terms.pickle")
regS_1.save(regression_results_dir + "/S_regression_all_events_no_site_terms.pickle")

#make prediciton and compare with the measured
y_P_predict_1 = regP_1.predict(peak_amplitude_df)
y_S_predict_1 = regS_1.predict(peak_amplitude_df)
# Compare Ground truth values
plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_1, y_S_predict_1, (1.0, 5.5), regression_results_dir + '/validate_predicted_strain_rate_all_events_no_site_terms.png')

# %% Regression 3: Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# first combined the nearby channels
for nearby_channel_number in [10, 20, 50, 100]:
    peak_amplitude_df_ridgecrest = combined_channels(DAS_index_ridgecrest, peak_amplitude_df_ridgecrest, nearby_channel_number)
    peak_amplitude_df_olancha = combined_channels(DAS_index_olancha, peak_amplitude_df_olancha, nearby_channel_number)

    peak_amplitude_df = pd.concat((peak_amplitude_df_ridgecrest, peak_amplitude_df_olancha), axis=0)
    # %% Aggregate the columns of region and combined_channel_id to form the regional site terms
    peak_amplitude_df['combined_channel_id']= peak_amplitude_df['combined_channel_id'].astype('str')
    peak_amplitude_df['region_site'] = peak_amplitude_df[['region', 'combined_channel_id']].agg('-'.join, axis=1)
    peak_amplitude_df = add_event_label(peak_amplitude_df)
    
    # Store the processed DataFrame
    peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)

    # %% Now can fit the data with different regional site terms
    regP_3 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df).fit()
    regS_3 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df).fit()

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP_3.params[-2:])
    print(regS_3.params[-2:])
    print('\n\n')   

    regP_3.save(regression_results_dir + f"/P_regression_region_site_terms_{nearby_channel_number}chan.pickle")
    regS_3.save(regression_results_dir + f"/S_regression_region_site_terms_{nearby_channel_number}chan.pickle")

        #make prediciton and compare with the measured
    y_P_predict_3 = regP_3.predict(peak_amplitude_df)
    y_S_predict_3 = regS_3.predict(peak_amplitude_df)

    plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_3, y_S_predict_3, (1.0, 5.5), 
    regression_results_dir + f'/validate_predicted_strain_rate_region_site_terms_{nearby_channel_number}chan.png')



# ======================= Below are the part to use the small events to do the regression ===========================
# %% Now only use the smaller earthquakes to do the regression
# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf_M3'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_region_site_all.csv')
peak_amplitude_df_M3 = peak_amplitude_df[peak_amplitude_df.magnitude < 4]

# %% Regression 1. Linear regression on the data point (this regression ignores the different site responses)
regP_1 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df_M3).fit()
regS_1 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df_M3).fit()

regP_1.save(regression_results_dir + "/P_regression_all_events_no_site_terms.pickle")
regS_1.save(regression_results_dir + "/S_regression_all_events_no_site_terms.pickle")

#make prediciton and compare with the measured
y_P_predict_1 = regP_1.predict(peak_amplitude_df)
y_S_predict_1 = regS_1.predict(peak_amplitude_df)
# Compare Ground truth values
plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_1, y_S_predict_1, (1.0, 5.5), regression_results_dir + '/validate_predicted_strain_rate_all_events_no_site_terms.png')

# %% Regression 3: Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# first combined the nearby channels
for nearby_channel_number in [10, 20, 50, 100]:

    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    peak_amplitude_df_M3 = peak_amplitude_df[peak_amplitude_df.magnitude < 4]

    # %% Now can fit the data with different regional site terms
    regP_3 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_M3).fit()
    regS_3 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_M3).fit()

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP_3.params[-2:])
    print(regS_3.params[-2:])
    print('\n\n')   

    regP_3.save(regression_results_dir + f"/P_regression_region_site_terms_{nearby_channel_number}chan.pickle")
    regS_3.save(regression_results_dir + f"/S_regression_region_site_terms_{nearby_channel_number}chan.pickle")

        #make prediciton and compare with the measured
    y_P_predict_3 = regP_3.predict(peak_amplitude_df)
    y_S_predict_3 = regS_3.predict(peak_amplitude_df)

    plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_3, y_S_predict_3, (1.0, 5.5), 
    regression_results_dir + f'/validate_predicted_strain_rate_region_site_terms_{nearby_channel_number}chan.png')


# %%
