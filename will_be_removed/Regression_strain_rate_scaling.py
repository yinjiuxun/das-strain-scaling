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

def filter_by_channel_number(peak_amplitude_df, min_channel):
    """To remove the measurements from few channels (< min_channel)"""
    event_channel_count = peak_amplitude_df.groupby(['event_id'])['event_id'].count()
    channel_count = event_channel_count.values
    event_id = event_channel_count.index
    event_id = event_id[channel_count >= min_channel]

    return peak_amplitude_df[peak_amplitude_df['event_id'].isin(event_id)]

def split_P_S_dataframe(peak_amplitude_df):
    # use P and S separately to do the regression
    peak_amplitude_df_P = peak_amplitude_df[['event_id', 'peak_P', 'magnitude', 'distance_in_km', 'combined_channel_id']]
    peak_amplitude_df_S = peak_amplitude_df[['event_id', 'peak_S', 'magnitude', 'distance_in_km', 'combined_channel_id']]

    # Remove some extreme data outliers before fitting
    peak_amplitude_df_P = peak_amplitude_df_P.dropna()
    peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.peak_P>0]
    peak_amplitude_df_P = peak_amplitude_df_P.drop(peak_amplitude_df_P[(peak_amplitude_df_P.peak_P > 1e3)].index)

    peak_amplitude_df_S = peak_amplitude_df_S.dropna()
    peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.peak_S>0]
    peak_amplitude_df_S = peak_amplitude_df_S.drop(peak_amplitude_df_S[(peak_amplitude_df_S.peak_S > 1e3)].index)

    return peak_amplitude_df_P, peak_amplitude_df_S

def fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, min_channel=None):
   
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)
 
    # Before regression, first filter events based on the channel number, if < 100, discard the event (probably a wrong pick/local noise)
    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
    regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()

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


def fit_regression_with_weight_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, min_channel=None):
    """ Linear regression with weight, the weight is 10**magnitude """

    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)

    # Before regression, first filter events based on the channel number, if < 100, discard the event (probably a wrong pick/local noise)
    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    regP = smf.wls(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', 
                data=peak_amplitude_df_P, weights = (10**peak_amplitude_df_P.magnitude)).fit()
    regS = smf.wls(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', 
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
    '''Regression including the distance attenuation'''
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)

    regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region):distance_in_km + C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
    regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region):distance_in_km + C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()

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
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events'
das_pick_file_name = 'calibrated_peak_amplitude.csv'
region_label = 'ridgecrest'


# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling'
das_pick_file_folder = '/peak_amplitude_events'
das_pick_file_name = 'peak_amplitude_M3+.csv'
region_label = 'olancha'

# ==============================  Mammoth data - South========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events'
das_pick_file_name = 'calibrated_peak_amplitude.csv'
region_label = 'mammothS'

# ==============================  Mammoth data - North========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events'
das_pick_file_name = 'calibrated_peak_amplitude.csv'
region_label = 'mammothN'

#%% load the peak amplitude results
# Load the peak amplitude results
snr_threshold = 10
min_channel = 100 # do regression only on events recorded by at least 100 channels
apply_calibrated_distance = True # if true, use the depth-calibrated distance to do regression

peak_amplitude_df, DAS_index = load_and_add_region(das_pick_file_folder + '/' + das_pick_file_name, 
                                                   region_label=region_label, snr_threshold=snr_threshold)

if apply_calibrated_distance: 
    peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']

# Preprocessing the peak amplitude data
#peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df = add_event_label(peak_amplitude_df)

#%% 
# Regression no attenuation
nearby_channel_number_list = [10, 20, 50, 100, -1]

regression_results_dir = results_output_dir + f'/regression_results_smf_{min_channel}_channel_at_least'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

regression_parameter_txt = regression_results_dir + '/regression_slopes'
mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

for nearby_channel_number in nearby_channel_number_list:
    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
    # Store the processed DataFrame
    if not os.path.exists(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv'):
        peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)
    # Specify magnitude range to do regression
    M_threshold = [2, 10]
    regP, regS = fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, min_channel=min_channel)
    
    mag_slopeP.append(regP.params[-2])
    dist_slopeP.append(regP.params[-1])
    mag_slopeS.append(regS.params[-2])
    dist_slopeS.append(regS.params[-1])
    
    # reset the regression models
    #del regP, regS

P_regression_parameter_df = pd.DataFrame({'site_channels':nearby_channel_number_list, 'magnitude-P':mag_slopeP, 'log(distance)-P':dist_slopeP})  
S_regression_parameter_df = pd.DataFrame({'site_channels':nearby_channel_number_list, 'magnitude-S':mag_slopeS, 'log(distance)-S':dist_slopeS})  
P_regression_parameter_df.to_csv(regression_parameter_txt + '_P.txt', index=False, sep='\t', float_format='%.3f')
S_regression_parameter_df.to_csv(regression_parameter_txt + '_S.txt', index=False, sep='\t', float_format='%.3f')

#%% 
# Regression with weight
nearby_channel_number_list = [10, 20, 50, 100, -1]

regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

regression_parameter_txt = regression_results_dir + '/regression_slopes'
mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

for nearby_channel_number in nearby_channel_number_list:
    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
    # Store the processed DataFrame
    if not os.path.exists(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv'):
        peak_amplitude_df.to_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv', index=False)
    # Specify magnitude range to do regression
    M_threshold = [2, 10]
    regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, min_channel=min_channel)
    
    mag_slopeP.append(regP.params[-2])
    dist_slopeP.append(regP.params[-1])
    mag_slopeS.append(regS.params[-2])
    dist_slopeS.append(regS.params[-1])
    
    # reset the regression models
    #del regP, regS

P_regression_parameter_df = pd.DataFrame({'site_channels':nearby_channel_number_list, 'magnitude-P':mag_slopeP, 'log(distance)-P':dist_slopeP})  
S_regression_parameter_df = pd.DataFrame({'site_channels':nearby_channel_number_list, 'magnitude-S':mag_slopeS, 'log(distance)-S':dist_slopeS})  
P_regression_parameter_df.to_csv(regression_parameter_txt + '_P.txt', index=False, sep='\t', float_format='%.3f')
S_regression_parameter_df.to_csv(regression_parameter_txt + '_S.txt', index=False, sep='\t', float_format='%.3f')

# %%
