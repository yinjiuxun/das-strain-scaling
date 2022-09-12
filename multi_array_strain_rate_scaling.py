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

# %% 
# Define Functions used here

def filter_by_channel_number(peak_amplitude_df, min_channel):
    """To remove the measurements from few channels (< min_channel)"""
    event_channel_count = peak_amplitude_df.groupby(['event_id'])['event_id'].count()
    channel_count = event_channel_count.values
    event_id = event_channel_count.index
    event_id = event_id[channel_count >= min_channel]

    return peak_amplitude_df[peak_amplitude_df['event_id'].isin(event_id)]

def split_P_S_dataframe(peak_amplitude_df, snr_threshold=None):
    # use P and S separately to do the regression
    peak_amplitude_df_P = peak_amplitude_df[['event_id', 'peak_P', 'magnitude', 'distance_in_km', 'region_site', 'snrP']]
    peak_amplitude_df_S = peak_amplitude_df[['event_id', 'peak_S', 'magnitude', 'distance_in_km', 'region_site', 'snrS']]

    # Remove some extreme data outliers before fitting
    peak_amplitude_df_P = peak_amplitude_df_P.dropna()
    peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.peak_P>0]
    peak_amplitude_df_P = peak_amplitude_df_P.drop(peak_amplitude_df_P[(peak_amplitude_df_P.peak_P > 1e3)].index)

    peak_amplitude_df_S = peak_amplitude_df_S.dropna()
    peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.peak_S>0]
    peak_amplitude_df_S = peak_amplitude_df_S.drop(peak_amplitude_df_S[(peak_amplitude_df_S.peak_S > 1e3)].index)

    if snr_threshold:
        peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.snrP >=snr_threshold]
        peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.snrS >=snr_threshold]

    return peak_amplitude_df_P, peak_amplitude_df_S

def model_parameters_df(reg, combined_channel_number, digits=3):
    magnitude = round(reg.params[-2],digits)
    distance = round(reg.params[-1],digits)
    magnitude_err = round(np.sqrt(reg.cov_params().iloc[-2][-2]),digits)
    distance_err = round(np.sqrt(reg.cov_params().iloc[-1][-1]),digits)
    parameter_df = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
    data = [[combined_channel_number, magnitude, distance, magnitude_err, distance_err]])
    return parameter_df

def write_regression_summary(regression_results_dir, file_name, reg):
    # make a directory to store the regression results in text
    regression_text = regression_results_dir + '/regression_results_txt'
    if not os.path.exists(regression_text):
        os.mkdir(regression_text)
    with open(regression_text + '/' + file_name + '.txt', "w") as text_file:
        text_file.write(reg.summary().as_text())

def fit_regression_magnitude_range(peak_amplitude_df, M_threshold, snr_threshold, regression_results_dir, nearby_channel_number, min_channel=None):
    
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df, snr_threshold)
    
    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_P).fit()
    regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_S).fit()

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

def fit_regression_with_weight_magnitude_range(peak_amplitude_df, M_threshold, snr_threshold, regression_results_dir, nearby_channel_number, min_channel=None):
    """ Linear regression with weight, the weight is 10**magnitude """

    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df, snr_threshold)

    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    regP = smf.wls(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', 
                data=peak_amplitude_df_P, weights = (10**peak_amplitude_df_P.magnitude)).fit()
    regS = smf.wls(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', 
                data=peak_amplitude_df_S, weights = (10**peak_amplitude_df_S.magnitude)).fit()

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
    '''Regression including the distance attenuation that is specific to the DAS array'''
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)

    regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region):distance_in_km + C(region_site) - 1', data=peak_amplitude_df_P).fit()
    regS = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region):distance_in_km + C(region_site) - 1', data=peak_amplitude_df_S).fit()

    print(regP.params[-6:])
    print(regS.params[-6:])
    print('\n\n')
    
    file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
    write_regression_summary(regression_results_dir, file_name_P, regP)
    file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"
    write_regression_summary(regression_results_dir, file_name_S, regS)

    regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
    regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
    return regP,regS

def secondary_site_calibration(regP, regS, peak_amplitude_df):
    y_P_predict = regP.predict(peak_amplitude_df)
    y_S_predict = regS.predict(peak_amplitude_df)

    peak_amplitude_df_temp = pd.DataFrame(columns=['region_site', 'channel_id', 'region', 'magnitude', 'diff_peak_P', 'diff_peak_S', 'site_calibate_P', 'site_calibrate_S'])
    peak_amplitude_df_temp.channel_id = peak_amplitude_df.channel_id
    peak_amplitude_df_temp.region = peak_amplitude_df.region
    peak_amplitude_df_temp.magnitude = peak_amplitude_df.magnitude

    #weighted_all = np.nansum(10**peak_amplitude_df.magnitude)
    peak_amplitude_df_temp.diff_peak_P = (-y_P_predict + np.log10(peak_amplitude_df.peak_P))#*10**peak_amplitude_df.magnitude/weighted_all
    peak_amplitude_df_temp.diff_peak_S = (-y_S_predict + np.log10(peak_amplitude_df.peak_S))#*10**peak_amplitude_df.magnitude/weighted_all
    peak_amplitude_df_temp.region_site = peak_amplitude_df_temp.region + '-' + peak_amplitude_df.combined_channel_id.astype('str')

    second_calibration = peak_amplitude_df_temp.groupby(['channel_id', 'region'], as_index=False).mean()
    temp_df = peak_amplitude_df_temp[['region_site', 'channel_id', 'region']].drop_duplicates(subset=['channel_id', 'region_site'])
    second_calibration = pd.merge(second_calibration, temp_df, on=['channel_id', 'region'])
    second_calibration = second_calibration.drop(columns=['magnitude'])
    return second_calibration


#%% 
peak_data_list = []
das_index_list = []
snr_threshold = 10
min_channel = 100 # do regression only on events recorded by at least 100 channels
magnitude_threshold = [2, 10]
combined_channel_number_list = [-1, 100, 50, 20, 10]#[10, 20, 50, 100, -1] # -1 means the constant model
apply_calibrated_distance = True # if true, use the depth-calibrated distance to do regression

# #%% ==============================  Ridgecrest data ========================================
# #ridgecrest_peaks = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_M3+.csv'
# ridgecrest_peaks = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events/calibrated_peak_amplitude.csv'
# peak_amplitude_df_ridgecrest, DAS_index_ridgecrest = load_and_add_region(ridgecrest_peaks, region_label='ridgecrest', 
#                                                                          snr_threshold=snr_threshold, magnitude_threshold=magnitude_threshold)
# peak_data_list.append(peak_amplitude_df_ridgecrest)
# das_index_list.append(DAS_index_ridgecrest)

# # #%% ==============================  Olancha data ========================================
# # #olancha_peaks = '/home/yinjx/kuafu/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_M3+.csv'
# # olancha_peaks = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling/peak_amplitude_M3+.csv'
# # peak_amplitude_df_olancha, DAS_index_olancha = load_and_add_region(olancha_peaks, region_label='olancha', snr_threshold=snr_threshold)
# # peak_data_list.append(peak_amplitude_df_olancha)
# # das_index_list.append(DAS_index_olancha)

# #%% ==============================  Mammoth south data ========================================
# mammoth_S_peaks = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events/calibrated_peak_amplitude.csv'
# peak_amplitude_df_mammoth_S, DAS_index_mammoth_S = load_and_add_region(mammoth_S_peaks, region_label='mammothS', 
#                                                                        snr_threshold=snr_threshold, magnitude_threshold=magnitude_threshold)
# peak_data_list.append(peak_amplitude_df_mammoth_S)
# das_index_list.append(DAS_index_mammoth_S)

# #%% ==============================  Mammoth north data ========================================
# mammoth_N_peaks = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events/calibrated_peak_amplitude.csv'
# peak_amplitude_df_mammoth_N, DAS_index_mammoth_N = load_and_add_region(mammoth_N_peaks, region_label='mammothN', 
#                                                                        snr_threshold=snr_threshold, magnitude_threshold=magnitude_threshold)
# peak_data_list.append(peak_amplitude_df_mammoth_N)
# das_index_list.append(DAS_index_mammoth_N)


#%% Combine the peak results from different regions
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
if not os.path.exists(results_output_dir):
    os.mkdir(results_output_dir)

# # #%% Preprocess the data file: combining different channels etc.
# # for nearby_channel_number in combined_channel_number_list:
# #     for ii, peak_data in enumerate(peak_data_list): # combine nearby channels for all the prepared data
# #         peak_data = combined_channels(das_index_list[ii], peak_data, nearby_channel_number)

# #     # Combined data from different regions
# #     peak_amplitude_df = pd.concat(peak_data_list, axis=0)
# #     peak_amplitude_df = add_event_label(peak_amplitude_df)

# #     if apply_calibrated_distance: 
# #         peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
    
# #     # %% Aggregate the columns of region and combined_channel_id to form the regional site terms
# #     peak_amplitude_df['combined_channel_id']= peak_amplitude_df['combined_channel_id'].astype('str')
# #     peak_amplitude_df['region_site'] = peak_amplitude_df[['region', 'combined_channel_id']].agg('-'.join, axis=1)

# #     # Store the processed DataFrame
# #     peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)

#%% 
# Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# directory to store the fitted results
regression_results_dir = results_output_dir + f'/regression_results_smf_{min_channel}_channel_at_least'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

regression_parameter_txt = regression_results_dir + '/regression_slopes'
mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

for nearby_channel_number in combined_channel_number_list:
    # Load the processed DataFrame
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    # Specify magnitude range to do regression
    M_threshold = [0, 9]
    snr_threshold = 10
    regP, regS = fit_regression_magnitude_range(peak_amplitude_df, M_threshold, snr_threshold, regression_results_dir, nearby_channel_number, min_channel=min_channel)

    #add secondary calibration second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df)
    second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df)
    second_calibration.to_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv', index=False)

    mag_slopeP.append(regP.params[-2])
    dist_slopeP.append(regP.params[-1])
    mag_slopeS.append(regS.params[-2])
    dist_slopeS.append(regS.params[-1])

P_regression_parameter_df = pd.DataFrame({'site_channels':combined_channel_number_list, 'magnitude-P':mag_slopeP, 'log(distance)-P':dist_slopeP})  
S_regression_parameter_df = pd.DataFrame({'site_channels':combined_channel_number_list, 'magnitude-S':mag_slopeS, 'log(distance)-S':dist_slopeS})  
P_regression_parameter_df.to_csv(regression_parameter_txt + '_P.txt', index=False, sep='\t', float_format='%.3f')
S_regression_parameter_df.to_csv(regression_parameter_txt + '_S.txt', index=False, sep='\t', float_format='%.3f')

#%% 
# Weighted Linear regression on the data point including the site term, here assume that every X nearby channels share the same site terms
# directory to store the fitted results
regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

regression_parameter_txt = regression_results_dir + '/regression_slopes'
mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

for nearby_channel_number in combined_channel_number_list:
    # Load the processed DataFrame
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    # Specify magnitude range to do regression
    M_threshold = [0, 9]
    snr_threshold = 10
    regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df, M_threshold, snr_threshold, regression_results_dir, nearby_channel_number, min_channel=min_channel)

    #add secondary calibration second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df)
    second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df)
    second_calibration.to_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv', index=False)

    mag_slopeP.append(regP.params[-2])
    dist_slopeP.append(regP.params[-1])
    mag_slopeS.append(regS.params[-2])
    dist_slopeS.append(regS.params[-1])

P_regression_parameter_df = pd.DataFrame({'site_channels':combined_channel_number_list, 'magnitude-P':mag_slopeP, 'log(distance)-P':dist_slopeP})  
S_regression_parameter_df = pd.DataFrame({'site_channels':combined_channel_number_list, 'magnitude-S':mag_slopeS, 'log(distance)-S':dist_slopeS})  
P_regression_parameter_df.to_csv(regression_parameter_txt + '_P.txt', index=False, sep='\t', float_format='%.3f')
S_regression_parameter_df.to_csv(regression_parameter_txt + '_S.txt', index=False, sep='\t', float_format='%.3f')


# %%
