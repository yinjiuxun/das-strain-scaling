#%%
from utility.general import mkdir
from utility.processing import filter_by_channel_number, split_P_S_dataframe
import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
import statsmodels.api as sm

def write_regression_summary(regression_results_dir, file_name, reg):
    # make a directory to store the regression results in text
    regression_text = regression_results_dir + '/regression_results_txt'
    mkdir(regression_text)
    with open(regression_text + '/' + file_name + '.txt', "w") as text_file:
        text_file.write(reg.summary().as_text())

def fit_regression(peak_amplitude_df, regression_results_dir, regression_results_filename, 
                                   weighted='ols', M_threshold=None, snr_threshold=None, min_channel=None):

    if M_threshold:
        peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df, snr_threshold)
    
    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    if weighted == 'ols': # Ordinary Linear Square (OLS)
        # Linear regression without weight
        regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_P).fit()
        regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_S).fit()
    
    elif weighted == 'wls': # Weighted Linear Square (WLS)
        # Linear regression with weight, the weight is 10**(magnitude/2)
        regP = smf.wls(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', 
                    data=peak_amplitude_df_P, weights = (10**peak_amplitude_df_P.magnitude)).fit()
        regS = smf.wls(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', 
                    data=peak_amplitude_df_S, weights = (10**peak_amplitude_df_S.magnitude)).fit() 
    else:
        raise TypeError(f'{weighted} is not defined, only "ols" or "wls"!')       

    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')
   
    # file_name_P = f"/P_{regression_results_filename}"
    # write_regression_summary(regression_results_dir, file_name_P, regP)
    # file_name_S = f"/S_{regression_results_filename}"
    # write_regression_summary(regression_results_dir, file_name_S, regS)

    # regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
    # regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
    return regP,regS

def store_regression_results(regX, regression_results_dir, results_filename):
    """Write regressio summary and store results"""
    write_regression_summary(regression_results_dir, results_filename, regX)
    regX.save(regression_results_dir + '/' + results_filename + '.pickle', remove_data=True)


def secondary_site_calibration(regP, regS, peak_amplitude_df):
    """Apply secondary calibration on the site terms"""
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