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

# ==============================  Ridgecrest data ========================================
#%% Specify the file names
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_name = '/peak_amplitude_M3+.csv'

# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/home/yinjx/kuafu/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_name = '/peak_amplitude_M3+.csv'

# ==============================  Mammoth data - South========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_name = '/Mammoth_South_Scaling_M3.csv'

# ==============================  Mammoth data - North========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_name = '/Mammoth_North_Scaling_M3.csv'

#%% load the peak amplitude results
# Load the peak amplitude results
peak_amplitude_df = pd.read_csv(results_output_dir + '/' + das_pick_file_name)

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

# make a directory to store the regression results in text
regression_text = regression_results_dir + '/regression_results_txt'
if not os.path.exists(regression_text):
    os.mkdir(regression_text)

peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df = add_event_label(peak_amplitude_df)
peak_amplitude_df.to_csv(results_output_dir + '/peak_amplitude_region_site_all.csv', index=False)
DAS_index = peak_amplitude_df.channel_id.unique().astype('int')

#%% Mammoth data contains the snrP and snrS, so if these two columns exist, only keep data with higher SNR
snr_threshold = 20
if 'snrP' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrP >= snr_threshold]

if 'snrS' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrS >= snr_threshold]

#%% Regression 1. Linear regression on the data point (this regression ignores the different site responses)
regP_1 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df).fit()
regS_1 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df).fit()

print(regP_1.summary())
print('\n\n')
print(regS_1.summary())

with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
    text_file.write(regP_1.summary().as_text())
with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
    text_file.write(regS_1.summary().as_text())

#make prediciton and compare with the measured
y_P_predict_1 = regP_1.predict(peak_amplitude_df)
y_S_predict_1 = regS_1.predict(peak_amplitude_df)

# Compare Ground truth values
plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_1, y_S_predict_1, (1.0, 5.5), regression_results_dir + '/validate_predicted_strain_rate_all_events_no_site_terms.png')

regP_1.save(regression_results_dir + "/P_regression_all_events_no_site_terms.pickle", remove_data=True)
regS_1.save(regression_results_dir + "/S_regression_all_events_no_site_terms.pickle", remove_data=True)

# %% Regression 2: Linear regression on the data point including the site term
# regP_2 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(channel_id) - 1', data=peak_amplitude_df).fit()
# regS_2 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(channel_id) - 1', data=peak_amplitude_df).fit()

# print(regP_2.params[-2:])
# print('\n\n')
# print(regS_2.params[-2:])

# regP_2.save(regression_results_dir + "/P_regression_all_events_with_site_terms.pickle")
# regS_2.save(regression_results_dir + "/S_regression_all_events_with_site_terms.pickle")

# #make prediciton and compare with the measured
# y_P_predict_2 = regP_2.predict(peak_amplitude_df)
# y_S_predict_2 = regS_2.predict(peak_amplitude_df)

# # Compare Ground truth values
# plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_2, y_S_predict_2, (1.0, 5.5), regression_results_dir + '/validate_predicted_strain_rate_all_events_with_site_terms.png')

# %% Regression 3: Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# nearby_channel_number = 50 # number of neighboring channels to have same site terms
for nearby_channel_number in [10, 20, 50, 100]:
    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
    # Store the processed DataFrame
    peak_amplitude_df = add_event_label(peak_amplitude_df)
    peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)


    regP_3 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()
    regS_3 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()

    print(regP_3.params[-2:])
    print(regS_3.params[-2:])
    print('\n\n')

    with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
        text_file.write(regP_3.summary().as_text())
    with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
        text_file.write(regS_3.summary().as_text())

    #make prediciton and compare with the measured
    y_P_predict_3 = regP_3.predict(peak_amplitude_df)
    y_S_predict_3 = regS_3.predict(peak_amplitude_df)

    plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_3, y_S_predict_3, (1.0, 5.5), 
    regression_results_dir + f'/validate_predicted_strain_rate_all_events_with_combined_site_terms_{nearby_channel_number}chan.png')

    regP_3.save(regression_results_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle", remove_data=True)
    regS_3.save(regression_results_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle", remove_data=True)

    # reset the regression models
    del regP_3, regS_3



# ======================= Below are the part to use the small events to do the regression ===========================
# %%
# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf_M4'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

# make a directory to store the regression results in text
regression_text = regression_results_dir + '/regression_results_txt'
if not os.path.exists(regression_text):
    os.mkdir(regression_text)

peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_region_site_all.csv')
peak_amplitude_df_M3 = peak_amplitude_df[peak_amplitude_df.magnitude < 4]

DAS_index = peak_amplitude_df_M3.channel_id.unique().astype('int')
#%% Mammoth data contains the snrP and snrS, so if these two columns exist, only keep data with higher SNR
snr_threshold = 20
if 'snrP' in peak_amplitude_df_M3.columns:
    peak_amplitude_df_M3 = peak_amplitude_df_M3[peak_amplitude_df_M3.snrP >= snr_threshold]

if 'snrS' in peak_amplitude_df_M3.columns:
    peak_amplitude_df_M3 = peak_amplitude_df_M3[peak_amplitude_df_M3.snrS >= snr_threshold]

#%% Regression 1. Linear regression on the data point (this regression ignores the different site responses)
regP_1 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df_M3).fit()
regS_1 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df_M3).fit()

print(regP_1.summary())
print('\n\n')
print(regS_1.summary())

with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
    text_file.write(regP_1.summary().as_text())
with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
    text_file.write(regS_1.summary().as_text())

regP_1.save(regression_results_dir + "/P_regression_all_events_no_site_terms.pickle")
regS_1.save(regression_results_dir + "/S_regression_all_events_no_site_terms.pickle")

#make prediciton and compare with the measured
y_P_predict_1 = regP_1.predict(peak_amplitude_df)
y_S_predict_1 = regS_1.predict(peak_amplitude_df)
plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_1, y_S_predict_1, (1.0, 5.5), regression_results_dir + '/validate_predicted_strain_rate_all_events_no_site_terms.png')

# Compare Ground truth values
# %% Regression 3: Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# nearby_channel_number = 50 # number of neighboring channels to have same site terms
for nearby_channel_number in [10, 20, 50, 100]:
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    peak_amplitude_df_M3 = peak_amplitude_df[peak_amplitude_df.magnitude < 4]

    regP_3 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_M3).fit()
    regS_3 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_M3).fit()

    print(regP_3.params[-2:])
    print(regS_3.params[-2:])
    print('\n\n')

    with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
        text_file.write(regP_3.summary().as_text())
    with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
        text_file.write(regS_3.summary().as_text())

    regP_3.save(regression_results_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS_3.save(regression_results_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    #make prediciton and compare with the measured
    y_P_predict_3 = regP_3.predict(peak_amplitude_df)
    y_S_predict_3 = regS_3.predict(peak_amplitude_df)

    plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_3, y_S_predict_3, (1.0, 5.5), 
    regression_results_dir + f'/validate_predicted_strain_rate_all_events_with_combined_site_terms_{nearby_channel_number}chan.png')

    # reset the regression models
    del regP_3, regS_3
# %%
