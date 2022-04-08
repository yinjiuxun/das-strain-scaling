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

def write_regression_summary(regression_results_dir, file_name, reg):
    # make a directory to store the regression results in text
    regression_text = regression_results_dir + '/regression_results_txt'
    if not os.path.exists(regression_text):
        os.mkdir(regression_text)
    with open(regression_text + '/' + file_name + '.txt', "w") as text_file:
        text_file.write(reg.summary().as_text())

def fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number):
   
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()
    regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()

    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')
    
    file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
    write_regression_summary(regression_results_dir, file_name_P, regP)
    file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"
    write_regression_summary(regression_results_dir, file_name_S, regS)

    regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
    regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
    return regP,regS

def fit_regression_with_attenuation_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number):
    '''Regression including the distance attenuation'''
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region):distance_in_km + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()
    regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region):distance_in_km + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()

    print(regP.params[-3:])
    print(regS.params[-3:])
    print('\n\n')
    
    file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
    write_regression_summary(regression_results_dir, file_name_P, regP)
    file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"
    write_regression_summary(regression_results_dir, file_name_S, regS)

    regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
    regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
    return regP,regS


# ==============================  Ridgecrest data ========================================
#%% Specify the file names
# results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
# das_pick_file_name = '/peak_amplitude_M3+.csv'
# region_label = 'ridgecrest'

results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate_refined'
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
peak_amplitude_df, DAS_index = load_and_add_region(results_output_dir + '/' + das_pick_file_name, 
                                                   region_label=region_label)

# Mammoth data contains the snrP and snrS, so if these two columns exist, only keep data with higher SNR
snr_threshold = 20
if 'snrP' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrP >= snr_threshold]

if 'snrS' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrS >= snr_threshold]

# Preprocessing the peak amplitude data
peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df = add_event_label(peak_amplitude_df)

#%% Regression no attenuation
nearby_channel_number_list = [10, 20, 50, 100, -1]

regression_results_dir = results_output_dir + '/regression_results_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

regression_parameter_txt = regression_results_dir + '/regression_slopes'
mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

for nearby_channel_number in nearby_channel_number_list:
    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
    # Store the processed DataFrame
    peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)
    # Specify magnitude range to do regression
    M_threshold = [0, 10]
    regP, regS = fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number)
    
    mag_slopeP.append(regP.params[-2])
    dist_slopeP.append(regP.params[-1])
    mag_slopeS.append(regS.params[-2])
    dist_slopeS.append(regS.params[-1])
    
    # reset the regression models
    del regP, regS

P_regression_parameter_df = pd.DataFrame({'site_channels':nearby_channel_number_list, 'magnitude-P':mag_slopeP, 'log(distance)-P':dist_slopeP})  
S_regression_parameter_df = pd.DataFrame({'site_channels':nearby_channel_number_list, 'magnitude-S':mag_slopeS, 'log(distance)-S':dist_slopeS})  
P_regression_parameter_df.to_csv(regression_parameter_txt + '_P.txt', index=False, sep='\t', float_format='%.3f')
S_regression_parameter_df.to_csv(regression_parameter_txt + '_S.txt', index=False, sep='\t', float_format='%.3f')

#%% ======================= Below are the part to use the small events to do the regression ===========================
nearby_channel_number_list = [10, 20, 50, 100, -1]
# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf_M4'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

regression_parameter_txt = regression_results_dir + '/regression_slopes'
mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

for nearby_channel_number in nearby_channel_number_list:
    # Load the processed DataFrame
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    # Specify magnitude range to do regression
    M_threshold = [0, 4]
    regP, regS = fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number)
    
    mag_slopeP.append(regP.params[-2])
    dist_slopeP.append(regP.params[-1])
    mag_slopeS.append(regS.params[-2])
    dist_slopeS.append(regS.params[-1])

    # reset the regression models
    del regP, regS

P_regression_parameter_df = pd.DataFrame({'site_channels':nearby_channel_number_list, 'magnitude-P':mag_slopeP, 'log(distance)-P':dist_slopeP})  
S_regression_parameter_df = pd.DataFrame({'site_channels':nearby_channel_number_list, 'magnitude-S':mag_slopeS, 'log(distance)-S':dist_slopeS})  
P_regression_parameter_df.to_csv(regression_parameter_txt + '_P.txt', index=False, sep='\t', float_format='%.3f')
S_regression_parameter_df.to_csv(regression_parameter_txt + '_S.txt', index=False, sep='\t', float_format='%.3f')






#%% ======================= Test other regression methods (NOT GOOD!) ===========================
dda
#%% Regression with attenuation (Ridgecrest and Mammoth North have issue: positive attenuation???!!!)
regression_results_dir = results_output_dir + '/regression_results_attenuation_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

for nearby_channel_number in [10, 20, 50, 100, -1]:
    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
    # Store the processed DataFrame
    peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)
    # Specify magnitude range to do regression
    M_threshold = [0, 10]
    regP, regS = fit_regression_with_attenuation_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number)
    # reset the regression models
    del regP, regS
#%% ===================== Including the attenuation terms ==============================
# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_attenuation_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

for nearby_channel_number in [10, 20, 50, 100, -1]:
    # Load the regression
    file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
    file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"

    regP = sm.load(regression_results_dir + '/' + file_name_P + '.pickle')
    regS = sm.load(regression_results_dir + '/' + file_name_S + '.pickle')

    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    
    y_P_predict = regP.predict(peak_amplitude_df)
    y_S_predict = regS.predict(peak_amplitude_df)
    
    plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict, y_S_predict, (1.0, 5.5), 
    regression_results_dir + f'/validate_predicted__combined_site_terms_{nearby_channel_number}chan.png')


#%%
regression_results_dir = results_output_dir + '/regression_results_attenuation_smf'
nearby_channel_number=-1
file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"

regP = sm.load(regression_results_dir + '/' + file_name_P + '.pickle')
regS = sm.load(regression_results_dir + '/' + file_name_S + '.pickle')

#%% Try the two-stage regression (Need to think again...)

regression_results_dir = results_output_dir + '/regression_results_smf_2_stages'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

nearby_channel_number = 20
M_threshold = [0, 10]


peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')

snr_threshold = 30
if 'snrP' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrP >= snr_threshold]

if 'snrS' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrS >= snr_threshold]

# Preprocessing the peak amplitude data
peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df = add_event_label(peak_amplitude_df)


peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
regP_stage1 = smf.ols(formula='np.log10(peak_P) ~ np.log10(distance_in_km) + distance_in_km + C(event_label) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit(method='pinv')

magnitude_df = pd.DataFrame(columns=['magnitude', 'magnitude_params'])
event_magnitude = peak_amplitude_df.groupby(by='event_label').mean().magnitude
magnitude_df.magnitude = event_magnitude
magnitude_df.magnitude_params = regP_stage1.params[:len(event_magnitude)].values

regP_stage2 = smf.ols(formula='magnitude_params ~ magnitude', data=magnitude_df).fit(method='pinv')

#%% Try mixed linear model 

regression_results_dir = results_output_dir + '/regression_results_smf_mlm'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

nearby_channel_number = 20
M_threshold = [0, 10]


peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')

snr_threshold = 30
if 'snrP' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrP >= snr_threshold]

if 'snrS' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.snrS >= snr_threshold]

# Preprocessing the peak amplitude data
peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df = add_event_label(peak_amplitude_df)


peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
regP = smf.mixedlm(formula='np.log10(peak_P) ~ np.log10(distance_in_km) + magnitude + C(combined_channel_id) - 1', 
data=peak_amplitude_df, groups=peak_amplitude_df['event_id']).fit()
print(regP.params)


# regP_stage2 = smf.ols(formula='np.log10(peak_P) ~ np.log10(distance_in_km) + distance_in_km + C(event_label) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit(method='qr')

# magnitude = peak_amplitude_df.groupby(by='event_label').mean().magnitude
# plt.plot(magnitude,regP_stage1.params[:230], 'x')
#%%
# # make a directory to store the regression results in text
# regression_text = regression_results_dir + '/regression_results_txt'
# if not os.path.exists(regression_text):
#     os.mkdir(regression_text)

# for nearby_channel_number in [10, 20, 50, 100, -1]:
#     # Load the processed DataFrame
#     peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
#     peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.magnitude < 4] # only use smaller events

#     regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()
#     regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()

#     print(regP.params[-2:])
#     print(regS.params[-2:])
#     print('\n\n')

#     with open(regression_text + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
#         text_file.write(regP.summary().as_text())
#     with open(regression_text + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
#         text_file.write(regS.summary().as_text())

#     regP.save(regression_results_dir + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.pickle", remove_data=True)
#     regS.save(regression_results_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle", remove_data=True)

#     # reset the regression models
#     del regP, regS


# #%%
# peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_region_site_all.csv')
# peak_amplitude_df_M3 = peak_amplitude_df[peak_amplitude_df.magnitude < 4]

# DAS_index = peak_amplitude_df_M3.channel_id.unique().astype('int')
# #%% Mammoth data contains the snrP and snrS, so if these two columns exist, only keep data with higher SNR
# snr_threshold = 20
# if 'snrP' in peak_amplitude_df_M3.columns:
#     peak_amplitude_df_M3 = peak_amplitude_df_M3[peak_amplitude_df_M3.snrP >= snr_threshold]

# if 'snrS' in peak_amplitude_df_M3.columns:
#     peak_amplitude_df_M3 = peak_amplitude_df_M3[peak_amplitude_df_M3.snrS >= snr_threshold]

# #%% Regression 1. Linear regression on the data point (this regression ignores the different site responses)
# regP_1 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df_M3).fit()
# regS_1 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df_M3).fit()

# print(regP_1.summary())
# print('\n\n')
# print(regS_1.summary())

# with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
#     text_file.write(regP_1.summary().as_text())
# with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
#     text_file.write(regS_1.summary().as_text())

# regP_1.save(regression_results_dir + "/P_regression_all_events_no_site_terms.pickle")
# regS_1.save(regression_results_dir + "/S_regression_all_events_no_site_terms.pickle")

# #make prediciton and compare with the measured
# y_P_predict_1 = regP_1.predict(peak_amplitude_df)
# y_S_predict_1 = regS_1.predict(peak_amplitude_df)
# plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_1, y_S_predict_1, (1.0, 5.5), regression_results_dir + '/validate_predicted_strain_rate_all_events_no_site_terms.png')

# # Compare Ground truth values
# # %% Regression 3: Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# # nearby_channel_number = 50 # number of neighboring channels to have same site terms
# for nearby_channel_number in [10, 20, 50, 100]:
#     peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
#     peak_amplitude_df_M3 = peak_amplitude_df[peak_amplitude_df.magnitude < 4]

#     regP_3 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_M3).fit()
#     regS_3 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_M3).fit()

#     print(regP_3.params[-2:])
#     print(regS_3.params[-2:])
#     print('\n\n')

#     with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
#         text_file.write(regP_3.summary().as_text())
#     with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
#         text_file.write(regS_3.summary().as_text())

#     regP_3.save(regression_results_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
#     regS_3.save(regression_results_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

#     #make prediciton and compare with the measured
#     y_P_predict_3 = regP_3.predict(peak_amplitude_df)
#     y_S_predict_3 = regS_3.predict(peak_amplitude_df)

#     plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_3, y_S_predict_3, (1.0, 5.5), 
#     regression_results_dir + f'/validate_predicted_strain_rate_all_events_with_combined_site_terms_{nearby_channel_number}chan.png')

#     # reset the regression models
#     del regP_3, regS_3
# # %%



# # #make prediciton and compare with the measured
# # y_P_predict_3 = regP_3.predict(peak_amplitude_df)
# # y_S_predict_3 = regS_3.predict(peak_amplitude_df)

# # plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_3, y_S_predict_3, (1.0, 5.5), 
# # regression_results_dir + f'/validate_predicted_strain_rate_all_events_with_combined_site_terms_{nearby_channel_number}chan.png')


# #%%

# def peak_amplitude_regression(peak_amplitude_df, regression_results_dir, regression_text):
#     regP_1 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df).fit()
#     regS_1 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df).fit()

#     print(regP_1.summary())
#     print('\n\n')
#     print(regS_1.summary())

#     with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
#         text_file.write(regP_1.summary().as_text())
#     with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
#         text_file.write(regS_1.summary().as_text())

# #make prediciton and compare with the measured
#     y_P_predict_1 = regP_1.predict(peak_amplitude_df)
#     y_S_predict_1 = regS_1.predict(peak_amplitude_df)

# # Compare Ground truth values
#     plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_1, y_S_predict_1, (1.0, 5.5), regression_results_dir + '/validate_predicted_strain_rate_all_events_no_site_terms.png')

#     regP_1.save(regression_results_dir + "/P_regression_all_events_no_site_terms.pickle", remove_data=True)
#     regS_1.save(regression_results_dir + "/S_regression_all_events_no_site_terms.pickle", remove_data=True)

# peak_amplitude_regression(peak_amplitude_df, regression_results_dir, regression_text)

# #%% Regression 1. Linear regression on the data point (this regression ignores the different site responses)
# regP_1 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df).fit()
# regS_1 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km)', data=peak_amplitude_df).fit()

# print(regP_1.summary())
# print('\n\n')
# print(regS_1.summary())

# with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
#     text_file.write(regP_1.summary().as_text())
# with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_-1chan.txt", "w") as text_file:
#     text_file.write(regS_1.summary().as_text())

# #make prediciton and compare with the measured
# y_P_predict_1 = regP_1.predict(peak_amplitude_df)
# y_S_predict_1 = regS_1.predict(peak_amplitude_df)

# # Compare Ground truth values
# plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_1, y_S_predict_1, (1.0, 5.5), regression_results_dir + '/validate_predicted_strain_rate_all_events_no_site_terms.png')

# regP_1.save(regression_results_dir + "/P_regression_all_events_no_site_terms.pickle", remove_data=True)
# regS_1.save(regression_results_dir + "/S_regression_all_events_no_site_terms.pickle", remove_data=True)

# # %% Regression 3: Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# # nearby_channel_number = 50 # number of neighboring channels to have same site terms
# for nearby_channel_number in [10, 20, 50, 100]:
#     peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
#     # Store the processed DataFrame
#     peak_amplitude_df = add_event_label(peak_amplitude_df)
#     peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)


#     regP_3 = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()
#     regS_3 = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df).fit()

#     print(regP_3.params[-2:])
#     print(regS_3.params[-2:])
#     print('\n\n')

#     with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
#         text_file.write(regP_3.summary().as_text())
#     with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
#         text_file.write(regS_3.summary().as_text())

#     #make prediciton and compare with the measured
#     y_P_predict_3 = regP_3.predict(peak_amplitude_df)
#     y_S_predict_3 = regS_3.predict(peak_amplitude_df)

#     plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict_3, y_S_predict_3, (1.0, 5.5), 
#     regression_results_dir + f'/validate_predicted_strain_rate_all_events_with_combined_site_terms_{nearby_channel_number}chan.png')

#     regP_3.save(regression_results_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle", remove_data=True)
#     regS_3.save(regression_results_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle", remove_data=True)

#     # reset the regression models
#     del regP_3, regS_3

