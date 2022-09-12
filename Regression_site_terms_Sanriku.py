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

import warnings
warnings.filterwarnings('ignore')

#%%
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

def fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None, min_channel=None):
   
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)
    
    # Before regression, first filter events based on the channel number, if < 100, discard the event (probably a wrong pick/local noise)
    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
    file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"
    regP, regS = None, None

    if given_coefficients is None:
        print('Fit for all coefficients........\n')
        if peak_amplitude_df_P.shape[0] != 0:
            regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
            print(regP.params[-2:])
            write_regression_summary(regression_results_dir, file_name_P, regP)
            regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)

        if peak_amplitude_df_S.shape[0] != 0:
            regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()      
            print(regS.params[-2:])
            write_regression_summary(regression_results_dir, file_name_S, regS)
            regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
        
    else: # given the coefficients from other array, fit for the site terms
        if (~np.isnan(given_coefficients[0])) & (~np.isnan(given_coefficients[1])):
            mag_coef_P, dist_coef_P = given_coefficients[0], given_coefficients[1]
            peak_amplitude_df_P.peak_P = peak_amplitude_df_P.peak_P / 10**(peak_amplitude_df_P.magnitude * mag_coef_P) / peak_amplitude_df_P.distance_in_km**dist_coef_P
            
            regP = smf.ols(formula='np.log10(peak_P) ~ C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
            regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
            #write_regression_summary(regression_results_dir, file_name_P, regP)

        if (~np.isnan(given_coefficients[2])) & (~np.isnan(given_coefficients[3])):
            mag_coef_S, dist_coef_S = given_coefficients[2], given_coefficients[3]
            peak_amplitude_df_S.peak_S = peak_amplitude_df_S.peak_S / 10**(peak_amplitude_df_S.magnitude * mag_coef_S) / peak_amplitude_df_S.distance_in_km**dist_coef_S
            
            regS = smf.ols(formula='np.log10(peak_S) ~ C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()
            regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
            #write_regression_summary(regression_results_dir, file_name_S, regS)
    
    return regP,regS


def fit_regression_with_weight_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None, min_channel=None):
    """ Linear regression with weight, the weight is 10**magnitude """

    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)

    # Before regression, first filter events based on the channel number, if < 100, discard the event (probably a wrong pick/local noise)
    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
    file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"
    regP, regS = None, None

    if given_coefficients is None:
        print('Fit for all coefficients........\n')
        if peak_amplitude_df_P.shape[0] != 0:
            regP = smf.wls(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', 
                        data=peak_amplitude_df_P, weights = (10**peak_amplitude_df_P.magnitude)).fit()
            print(regP.params[-2:])
            write_regression_summary(regression_results_dir, file_name_P, regP)
            regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)

        if peak_amplitude_df_S.shape[0] != 0:
            regS = smf.wls(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', 
                        data=peak_amplitude_df_S, weights = (10**peak_amplitude_df_S.magnitude)).fit()
            print(regS.params[-2:])
            write_regression_summary(regression_results_dir, file_name_S, regS)
            regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)

    else: # given the coefficients from other array, fit for the site terms
        if (~np.isnan(given_coefficients[0])) & (~np.isnan(given_coefficients[1])):
            mag_coef_P, dist_coef_P = given_coefficients[0], given_coefficients[1]
            peak_amplitude_df_P.peak_P = peak_amplitude_df_P.peak_P / 10**(peak_amplitude_df_P.magnitude * mag_coef_P) / peak_amplitude_df_P.distance_in_km**dist_coef_P
            
            regP = smf.ols(formula='np.log10(peak_P) ~ C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
            regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
            #write_regression_summary(regression_results_dir, file_name_P, regP)

        if (~np.isnan(given_coefficients[2])) & (~np.isnan(given_coefficients[3])):
            mag_coef_S, dist_coef_S = given_coefficients[2], given_coefficients[3]
            peak_amplitude_df_S.peak_S = peak_amplitude_df_S.peak_S / 10**(peak_amplitude_df_S.magnitude * mag_coef_S) / peak_amplitude_df_S.distance_in_km**dist_coef_S
            
            regS = smf.ols(formula='np.log10(peak_S) ~ C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()
            regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
            #write_regression_summary(regression_results_dir, file_name_S, regS)
            
    return regP,regS

def secondary_site_calibration(regP, regS, peak_amplitude_df, given_coefficients=None):
    """obtain secondary site term calibration"""

    peak_amplitude_df_temp = pd.DataFrame(columns=['region_site', 'channel_id', 'region', 'magnitude', 'diff_peak_P', 'diff_peak_S', 'site_calibate_P', 'site_calibrate_S'])
    peak_amplitude_df_temp.channel_id = peak_amplitude_df.channel_id
    peak_amplitude_df_temp.region = peak_amplitude_df.region
    peak_amplitude_df_temp.magnitude = peak_amplitude_df.magnitude

    if regP is None:
        peak_amplitude_df_temp.diff_peak_P = np.nan
    else:
        y_P_predict = regP.predict(peak_amplitude_df)
        if given_coefficients:
            mag_coef_P, dist_coef_P = given_coefficients[0], given_coefficients[1]
            y_P_predict = y_P_predict + (peak_amplitude_df.magnitude * mag_coef_P) + dist_coef_P * np.log10(peak_amplitude_df.distance_in_km)

        peak_amplitude_df_temp.diff_peak_P = (-y_P_predict + np.log10(peak_amplitude_df.peak_P))#*10**peak_amplitude_df.magnitude/weighted_all
    
    if regS is None:
        peak_amplitude_df_temp.diff_peak_S = np.nan
    else:
        y_S_predict = regS.predict(peak_amplitude_df)
        if given_coefficients:
            mag_coef_S, dist_coef_S = given_coefficients[2], given_coefficients[3]
            y_S_predict = y_S_predict + peak_amplitude_df.magnitude * mag_coef_S + dist_coef_S * np.log10(peak_amplitude_df.distance_in_km)
        
        peak_amplitude_df_temp.diff_peak_S = (-y_S_predict + np.log10(peak_amplitude_df.peak_S))#*10**peak_amplitude_df.magnitude/weighted_all
        
    peak_amplitude_df_temp.region_site = peak_amplitude_df_temp.region + '-' + peak_amplitude_df.combined_channel_id.astype('str')

    second_calibration = peak_amplitude_df_temp.groupby(['channel_id', 'region'], as_index=False).mean()
    temp_df = peak_amplitude_df_temp[['region_site', 'channel_id', 'region']].drop_duplicates(subset=['channel_id', 'region_site'])
    second_calibration = pd.merge(second_calibration, temp_df, on=['channel_id', 'region'])
    second_calibration = second_calibration.drop(columns=['magnitude'])
    return second_calibration

def calculate_magnitude_from_strain(peak_amplitude_df, reg, type, site_term_column='region_site', given_coefficients=None, secondary_calibration=None):
    
    if secondary_calibration:
        second_calibration = pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv')
        peak_amplitude_df = pd.merge(peak_amplitude_df, second_calibration, on=['channel_id', 'region'])
        # MUST BE VERY CAREFUL WHEN USING merge, it changes the order of DataFrame!

    # get the annoying categorical keys
    try:
        site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
    except:
        raise NameError(f"Index {site_term_column} doesn't exist!")

    if given_coefficients is None:
        mag_coef, dist_coef = reg.params['magnitude'], reg.params['np.log10(distance_in_km)']
    else:
        mag_coef, dist_coef = given_coefficients[0], given_coefficients[1]

    if type == 'P':
        M_predict = (np.log10(peak_amplitude_df.peak_P) \
                    - np.array(reg.params[site_term_keys]) \
                    - dist_coef * np.log10(peak_amplitude_df.distance_in_km)) \
                    / mag_coef
        if secondary_calibration:
                M_predict = M_predict.copy() - peak_amplitude_df.diff_peak_P/mag_coef

    elif type == 'S':
        M_predict = (np.log10(peak_amplitude_df.peak_S) \
                    - np.array(reg.params[site_term_keys]) \
                    - dist_coef * np.log10(peak_amplitude_df.distance_in_km)) \
                    / mag_coef
        if secondary_calibration:
                M_predict = M_predict.copy() - peak_amplitude_df.diff_peak_S/mag_coef

    return M_predict, peak_amplitude_df

def get_mean_magnitude(peak_amplitude_df, M_predict):
    temp_df = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df['predicted_M'] = M_predict
    temp_df = temp_df.groupby(temp_df['event_id']).aggregate('mean')

    temp_df2 = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df2['predicted_M_std'] = M_predict
    temp_df2 = temp_df2.groupby(temp_df2['event_id']).aggregate(np.std)

    temp_df = pd.concat([temp_df, temp_df2['predicted_M_std']], axis=1)
    return temp_df

def store_site_terms(regression_results_dir, combined_channel_number, peak_amplitude_df_fit):
    """Function to extract site terms as individual csv file"""
    temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
    temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])

    try:
        regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
        site_term_P = regP.params[:-2]
        # temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
        temp_df_P['combined_channel_id'] = [int(temp.replace('C(combined_channel_id)[', '').replace(']', '')) for temp in site_term_P.index]
        temp_df_P['site_term_P'] = np.array(site_term_P)
    except:
        print('P regression not found, assign Nan to the site term')
        site_term_P = np.nan
    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
        site_term_S = regS.params[:-2]
        # temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])
        temp_df_S['combined_channel_id'] = [int(temp.replace('C(combined_channel_id)[', '').replace(']', '')) for temp in site_term_S.index]
        temp_df_S['site_term_S'] = np.array(site_term_S)
    except:
        print('S regression not found, assign Nan to the site term')
        site_term_S = np.nan

    peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, combined_channel_number)
    combined_channel_id = np.sort(peak_amplitude_df_fit.combined_channel_id.unique())

    # combining the site terms into one csv file
    temp_df1 = pd.DataFrame(columns=['combined_channel_id'])
    temp_df1['combined_channel_id'] = combined_channel_id
    temp_df1 = pd.merge(temp_df1, temp_df_P, on='combined_channel_id', how='outer')
    temp_df1 = pd.merge(temp_df1, temp_df_S, on='combined_channel_id', how='outer')

    temp_df2 = peak_amplitude_df_fit.loc[:, ['combined_channel_id', 'channel_id']]
    temp_df2 = temp_df2.drop_duplicates(subset=['channel_id']).sort_values(by='channel_id')
    site_term_df = pd.merge(temp_df2, temp_df1, on='combined_channel_id')

    # load secondary calibration and merge
    second_calibration = pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv')
    site_term_df = pd.merge(site_term_df, second_calibration, on='channel_id')
    
    # Store the site terms
    site_term_df.to_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv', index=False)

    return site_term_df

# %% ==========================================================================================
# Apply to more data to predict the magnitude
import seaborn as sns
def plot_magnitude_seaborn(df_magnitude):
    sns.set_theme(style="ticks", font_scale=2)

    g = sns.JointGrid(data=df_magnitude, x="magnitude", y="predicted_M", marginal_ticks=True,
    xlim=(0, 7), ylim=(0, 7), height=10, space=0.3)


    # Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

    # Add the joint and marginal histogram plots 03012d
    g.plot_joint(
    sns.histplot, discrete=(False, False),
    cmap="light:#9C4D4D", pmax=.7, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", color="#9C4D4D") #flare; light:#9C4D4D; dark:salmon_r

    g.ax_joint.plot([0,8], [0,8], 'k-', linewidth = 2)
    g.ax_joint.plot([0,8], [1,9], 'k--', linewidth = 1)
    g.ax_joint.plot([0,8], [-1,7], 'k--', linewidth = 1)
    g.ax_joint.set_xlabel('catalog magnitude')
    g.ax_joint.set_ylabel('predicted magnitude')

    return g

def plot_prediction_vs_measure_seaborn(peak_comparison_df, xy_range, phase):
    sns.set_theme(style="ticks", font_scale=2)
    if phase == 'P':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_P", y="peak_P_predict", marginal_ticks=True,
                        xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    elif phase == 'S':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_S", y="peak_S_predict", marginal_ticks=True,
                        xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    g.ax_joint.set(xscale="log")
    g.ax_joint.set(yscale="log")

    # Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

    # Add the joint and marginal histogram plots 03012d
    bins=40
    g.plot_joint(sns.histplot, discrete=(False, False), bins=(bins, bins), cmap="light:#4D4D9C", pmax=.6, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", bins=bins, binrange=(-2, 2), color="#4D4D9C")

    g.ax_joint.plot(xy_range, xy_range, 'k-', linewidth = 2)
    g.ax_joint.set_xlabel('measured peak strain rate\n (micro strain/s)')
    g.ax_joint.set_ylabel('calculated peak strain rate\n (micro strain/s)')
    return g

# ==============================  Sanriku data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events'
das_pick_file_name = 'calibrated_peak_amplitude.csv'
region_label = 'sanriku'
min_channel = 100
apply_calibrated_distance = True
#%% load the peak amplitude results
# Load the peak amplitude results for regression
snr_threshold = 10
M_threshold = [2, 10]
peak_amplitude_df_fit, DAS_index = load_and_add_region(das_pick_file_folder + '/' + das_pick_file_name, 
                                                   region_label=region_label, snr_threshold=snr_threshold)

if apply_calibrated_distance:
    peak_amplitude_df_fit['distance_in_km'] = peak_amplitude_df_fit['calibrated_distance_in_km']

peak_amplitude_df_fit = peak_amplitude_df_fit[(peak_amplitude_df_fit.magnitude >= M_threshold[0]) & 
                                              (peak_amplitude_df_fit.magnitude <= M_threshold[1]) &
                                              (peak_amplitude_df_fit.QA == 'Yes')]
# filter by the channel number
peak_amplitude_df_fit = filter_by_channel_number(peak_amplitude_df_fit, min_channel)

# Preprocessing the peak amplitude data
#peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df_fit = add_event_label(peak_amplitude_df_fit)

# Drop 4130
peak_amplitude_df_fit = peak_amplitude_df_fit.drop(peak_amplitude_df_fit[peak_amplitude_df_fit.event_id==4130].index)

# Randomly select 10 only 
random_id = np.random.choice(peak_amplitude_df_fit.event_id.unique(), 10, replace=False)
peak_amplitude_df_fit = peak_amplitude_df_fit[peak_amplitude_df_fit.event_id.isin(random_id)]

fit_data_info_str = f'================ {len(peak_amplitude_df_fit.event_id.unique())} events used to constrain site terms ============\n' + \
                    f'SNR >= {snr_threshold} \n' + \
                    f'Magnitude: {M_threshold} \n' + \
                    f'{peak_amplitude_df_fit.shape[0]} measurements \n\n'
print(fit_data_info_str)

# %% ==========================================================================================
# Use the site term to estimate magnitude
# Load the peak amplitude results
snr_threshold = 5
M_threshold = [1, 10]

peak_amplitude_df_predict, DAS_index = load_and_add_region(das_pick_file_folder + '/' + das_pick_file_name, 
                                                   region_label=region_label, snr_threshold=snr_threshold)

# if apply_calibrated_distance:
#     peak_amplitude_df_predict['distance_in_km'] = peak_amplitude_df_predict['calibrated_distance_in_km']

# Preprocessing the peak amplitude data
#peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df_predict = add_event_label(peak_amplitude_df_predict)

peak_amplitude_df_predict = peak_amplitude_df_predict.drop(index=peak_amplitude_df_fit.index)
      
peak_amplitude_df_predict = peak_amplitude_df_predict[(peak_amplitude_df_predict.magnitude >= M_threshold[0]) & 
                                                      (peak_amplitude_df_predict.magnitude <= M_threshold[1]) & 
                                                      (peak_amplitude_df_predict.QA == 'Yes')]

# filter by the channel number
peak_amplitude_df_predict = filter_by_channel_number(peak_amplitude_df_predict, 50)

predict_data_info_str = f'================ {len(peak_amplitude_df_predict.event_id.unique())} events for validation =============\n' + \
                        f'SNR >= {snr_threshold} \n' + \
                        f'Magnitude: {M_threshold} \n' + \
                        f'{peak_amplitude_df_predict.shape[0]} measurements \n\n'
print(predict_data_info_str)

#%% ==========================================================================================
# write some data info
with open(results_output_dir + '/data_info.txt', 'w') as f:
    f.write(fit_data_info_str)
    f.write(predict_data_info_str)

#%% ==========================================================================================
# Fit the site term only
# Regression OLS
if region_label == 'sanriku':
    given_coefficients_list = [[np.nan, np.nan, 0.649, -1.372],
                               [np.nan, np.nan, 0.653, -1.382],
                               [np.nan, np.nan, 0.661, -1.401],
                               [np.nan, np.nan, 0.667, -1.417]]

    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + f'/regression_results_smf_{min_channel}_channel_at_least'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
        given_coefficients = given_coefficients_list[ii]
        regP, regS = fit_regression_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, 
                    nearby_channel_number, given_coefficients=given_coefficients)

        second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df_fit, given_coefficients=given_coefficients)
        second_calibration.to_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv', index=False)


#%% ==========================================================================================
# Regression with weight
if region_label == 'sanriku':
    given_coefficients_list = [[np.nan, np.nan, 0.590, -1.304],
                               [np.nan, np.nan, 0.595, -1.320],
                               [np.nan, np.nan, 0.608, -1.358],
                               [np.nan, np.nan, 0.619, -1.393]]

    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
        given_coefficients = given_coefficients_list[ii]
        regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, 
                    nearby_channel_number, given_coefficients=given_coefficients)

        second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df_fit, given_coefficients=given_coefficients)
        second_calibration.to_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv', index=False)


# %% ==========================================================================================
# Output site terms
# regression_results_dir = results_output_dir + '/regression_results_smf_weighted'
regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
nearby_channel_number_list = [10, 20, 50, 100]
for combined_channel_number in nearby_channel_number_list:
    store_site_terms(regression_results_dir, combined_channel_number, peak_amplitude_df_fit)

    # temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
    # temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])

    # try:
    #     regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    #     site_term_P = regP.params[:-2]
    #     # temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
    #     temp_df_P['combined_channel_id'] = [int(temp.replace('C(combined_channel_id)[', '').replace(']', '')) for temp in site_term_P.index]
    #     temp_df_P['site_term_P'] = np.array(site_term_P)
    # except:
    #     print('P regression not found, assign Nan to the site term')
    #     site_term_P = np.nan
    # try:
    #     regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    #     site_term_S = regS.params[:-2]
    #     # temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])
    #     temp_df_S['combined_channel_id'] = [int(temp.replace('C(combined_channel_id)[', '').replace(']', '')) for temp in site_term_S.index]
    #     temp_df_S['site_term_S'] = np.array(site_term_S)
    # except:
    #     print('S regression not found, assign Nan to the site term')
    #     site_term_S = np.nan

    # peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, combined_channel_number)
    # combined_channel_id = np.sort(peak_amplitude_df_fit.combined_channel_id.unique())

    # # combining the site terms into one csv file
    # temp_df1 = pd.DataFrame(columns=['combined_channel_id'])
    # temp_df1['combined_channel_id'] = combined_channel_id
    # temp_df1 = pd.merge(temp_df1, temp_df_P, on='combined_channel_id', how='outer')
    # temp_df1 = pd.merge(temp_df1, temp_df_S, on='combined_channel_id', how='outer')

    # temp_df2 = peak_amplitude_df_fit.loc[:, ['combined_channel_id', 'channel_id']]
    # temp_df2 = temp_df2.drop_duplicates(subset=['channel_id']).sort_values(by='channel_id')
    # site_term_df = pd.merge(temp_df2, temp_df1, on='combined_channel_id')

    # # load secondary calibration and merge
    # second_calibration = pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv')
    # site_term_df = pd.merge(site_term_df, second_calibration, on='channel_id')
    
    # # Store the site terms
    # site_term_df.to_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv', index=False)


#%% ==========================================================================================
# check site terms without calibration
plt.close('all')
fig, ax = plt.subplots(2,1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
gca = ax[0]
gca.hist(peak_amplitude_df_fit.channel_id*5/1e3, range=(0.5*5/1e3, 10000.5*5/1e3), bins=50)
gca.set_ylabel('Counts')
gca.set_title('Number of measurements')
gca.grid()

zorder_num = 0
gca = ax[1]
for i_model, combined_channel_number in enumerate(nearby_channel_number_list):
    try:
        site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')
        # gca.plot(site_term_df.channel_id, site_term_df.site_term_P, '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
        gca.plot(site_term_df.channel_id*5/1e3, site_term_df.site_term_S, '-', label=f'{combined_channel_number} channels', zorder=i_model)#, Cond.# {regS.condition_number:.2f}')
    except:
        continue
    # reset the regression models
    #del regP, regS

# gca.set_yscale('log')
gca.set_ylabel('Site terms S (in log10)')
gca.set_xlabel('Channel numbers')
gca.grid()

#gca.set_xticklabels(np.arange(0, 10000, 1000))
gca.legend(loc=3, fontsize=14)
plt.savefig(regression_results_dir + '/site_terms.png', bbox_inches='tight')
plt.savefig(regression_results_dir + '/site_terms.pdf', bbox_inches='tight')

# %% ==========================================================================================
# Check site terms with calibration
plt.close('all')
fig, ax = plt.subplots(2,1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
gca = ax[0]
gca.hist(peak_amplitude_df_fit.channel_id*5/1e3, range=(0.5*5/1e3, 10000.5*5/1e3), bins=50)
gca.set_ylabel('Counts')
gca.set_title('Number of measurements')
gca.grid()

zorder_num = 0
gca = ax[1]
for i_model, combined_channel_number in enumerate(nearby_channel_number_list):
    try:
        site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')
        # gca.plot(site_term_df.channel_id, site_term_df.site_term_P, '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
        gca.plot(site_term_df.channel_id*5/1e3, site_term_df.site_term_S + site_term_df.diff_peak_S, '-', label=f'{combined_channel_number} channels', zorder=i_model)#, Cond.# {regS.condition_number:.2f}')
    except:
        continue
    # reset the regression models
    #del regP, regS

# gca.set_yscale('log')
gca.set_ylabel('Site terms S (in log10)')
gca.set_xlabel('Channel numbers')
gca.grid()
#gca.set_xlim(10, 12)

#gca.set_xticklabels(np.arange(0, 10000, 1000))
gca.legend(loc=3, fontsize=14)
plt.savefig(regression_results_dir + '/site_terms_calibrated.png', bbox_inches='tight')
plt.savefig(regression_results_dir + '/site_terms_calibrated.pdf', bbox_inches='tight')



#%% 
# =====================magnitude prediction================================
weighted = 'weighted_'
secondary_calibration = True

for weighted in ['', 'weighted_']:
    for secondary_calibration in [True, False]:

        regression_results_dir = results_output_dir + f'/regression_results_smf_{weighted}{min_channel}_channel_at_least'
        # regression_results_dir = results_output_dir + '/regression_results_smf_weighted'

        if secondary_calibration:
            calibrated_label = '_calibrated'
        else:
            calibrated_label = ''

        if weighted == '':
            given_coefficients_list = [[0.649, -1.372],
                                    [0.653, -1.382],
                                    [0.661, -1.401],
                                    [0.667, -1.417]] # coefficients of unweighted regression
        elif weighted == 'weighted_':                            
            given_coefficients_list = [[0.590, -1.304],
                                    [0.595, -1.320],
                                    [0.608, -1.358],
                                    [0.619, -1.393]]  # coefficients of weighted regression
        else:
            raise NameError            

        # given_coefficients_list = [None, None, None, None]

        # do the magnitude estimation with the regression results
        nearby_channel_number_list = [100, 50, 20, 10]
        for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
            peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
            peak_amplitude_df_predict = combined_channels(DAS_index, peak_amplitude_df_predict, nearby_channel_number)

            regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
            
            # only keep the channels that have been constrained
            site_term_column='combined_channel_id'
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df_predict[site_term_column]])
            ii_channel = np.isin(site_term_keys, regS.params.keys())
            peak_amplitude_df_predict = peak_amplitude_df_predict[ii_channel]

            # Apply to the fit set
            M_fit, peak_amplitude_df_fit_temp = calculate_magnitude_from_strain(peak_amplitude_df_fit, regS, type='S', 
                                site_term_column=site_term_column, given_coefficients=given_coefficients_list[ii], secondary_calibration=secondary_calibration)
            M_df_fit = get_mean_magnitude(peak_amplitude_df_fit_temp, M_fit)

            # Apply to the prediction set
            M_predict, peak_amplitude_df_predict_temp = calculate_magnitude_from_strain(peak_amplitude_df_predict, regS, type='S', 
                                site_term_column=site_term_column, given_coefficients=given_coefficients_list[ii], secondary_calibration=secondary_calibration)
            M_df_predict = get_mean_magnitude(peak_amplitude_df_predict_temp, M_predict)

            gca = plot_magnitude_seaborn(M_df_predict)
            gca.ax_joint.plot(M_df_fit.magnitude, M_df_fit.predicted_M, 'k.', markersize=10)
            if 4130 in M_df_fit.index:
                gca.ax_joint.plot(M_df_fit[M_df_fit.index == 4130].magnitude, M_df_fit[M_df_fit.index == 4130].predicted_M, 'r.', markersize=15)
                gca.ax_joint.text(M_df_fit[M_df_fit.index == 4130].magnitude, M_df_fit[M_df_fit.index == 4130].predicted_M+0.2, 'clipped', color='r')
            gca.savefig(regression_results_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn{calibrated_label}.png",bbox_inches='tight')
            gca.savefig(regression_results_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn{calibrated_label}.pdf",bbox_inches='tight')
            plt.close('all')

# %%Validate the strain rate for all data using the transfered scaling
import seaborn as sns
peak_amplitude_df_all0 = pd.concat([peak_amplitude_df_fit, peak_amplitude_df_predict], axis=0)

weighted = 'weighted_'
secondary_calibration = True

for weighted in ['', 'weighted_']:
    for secondary_calibration in [True, False]:

        if secondary_calibration:
            calibrated_label = '_calibrated'
        else:
            calibrated_label = ''

        if_given_coefficents = True
        nearby_channel_number_list = [100, 50, 20, 10]


        regression_results_dir = results_output_dir + f'/regression_results_smf_{weighted}{min_channel}_channel_at_least'
        # regression_results_dir = results_output_dir + '/regression_results_smf_weighted'

        if weighted == '':
            given_coefficients_list = [[0.649, -1.372],
                                    [0.653, -1.382],
                                    [0.661, -1.401],
                                    [0.667, -1.417]] # coefficients of unweighted regression
        elif weighted == 'weighted_':                            
            given_coefficients_list = [[0.590, -1.304],
                                    [0.595, -1.320],
                                    [0.608, -1.358],
                                    [0.619, -1.393]]  # coefficients of weighted regression
        else:
            raise NameError      

                            
        plt.close('all')
        for ii, combined_channel_number in enumerate(nearby_channel_number_list):

            given_coefficients = given_coefficients_list[ii]
            mag_coef_S, dist_coef_S = given_coefficients[0], given_coefficients[1]

            regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
            
            peak_amplitude_df_all = combined_channels(DAS_index, peak_amplitude_df_all0, combined_channel_number)

            # only keep the channels that have been constrained
            site_term_column='combined_channel_id'
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df_all[site_term_column]])
            ii_channel = np.isin(site_term_keys, regS.params.keys())
            peak_amplitude_df_all = peak_amplitude_df_all[ii_channel]

            if secondary_calibration:
                second_calibration = pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{combined_channel_number}chan.csv')
                peak_amplitude_df_all = pd.merge(peak_amplitude_df_all, second_calibration, on=['channel_id', 'region'])
            
            peak_amplitude_calculated = regS.predict(peak_amplitude_df_all)

            if if_given_coefficents:
                peak_amplitude_calculated = peak_amplitude_calculated + (peak_amplitude_df_all.magnitude * mag_coef_S) \
                                            + np.log10(peak_amplitude_df_all.distance_in_km)*dist_coef_S

            if secondary_calibration:
                peak_amplitude_calculated = peak_amplitude_calculated + peak_amplitude_df_all.diff_peak_S

            peak_amplitude_calculated = 10**peak_amplitude_calculated


            temp_peaks = np.array([np.array(peak_amplitude_df_all.peak_S), peak_amplitude_calculated]).T
            peak_comparison_df = pd.DataFrame(data=temp_peaks,
                                        columns=['peak_S', 'peak_S_predict'])

            g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S')
            g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn{calibrated_label}.png')
            g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn{calibrated_label}.pdf')
            plt.close('all')



# %%
# Directly fit all available data without coefficient-transferring
peak_amplitude_df_all = pd.concat([peak_amplitude_df_fit, peak_amplitude_df_predict], axis=0)

# Regression without weight, drop the event 4130
peak_amplitude_df_all = peak_amplitude_df_all.drop(index = peak_amplitude_df_all[peak_amplitude_df_all.event_id == 4130].index)

if region_label == 'sanriku':
    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + f'/regression_results_smf_all_coefficients_drop_4130_{min_channel}_channel_at_least'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_all = combined_channels(DAS_index, peak_amplitude_df_all, nearby_channel_number)
        regP, regS = fit_regression_magnitude_range(peak_amplitude_df_all, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)
        # apply secondary calibration
        second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df_all)
        second_calibration.to_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv', index=False)

    # output site terms
    for combined_channel_number in nearby_channel_number_list:
        store_site_terms(regression_results_dir, combined_channel_number, peak_amplitude_df_all)


# Regression with weight, drop the event 4130
peak_amplitude_df_all = peak_amplitude_df_all.drop(index = peak_amplitude_df_all[peak_amplitude_df_all.event_id == 4130].index)
if region_label == 'sanriku':
    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_all_coefficients_drop_4130_{min_channel}_channel_at_least'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_all = combined_channels(DAS_index, peak_amplitude_df_all, nearby_channel_number)
        regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df_all, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)
        # apply secondary calibration
        second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df_all)
        second_calibration.to_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv', index=False)

    # output site terms
    for combined_channel_number in nearby_channel_number_list:
        store_site_terms(regression_results_dir, combined_channel_number, peak_amplitude_df_all)


#%% plot site terms
regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_all_coefficients_drop_4130_{min_channel}_channel_at_least'
# Check site terms with calibration
plt.close('all')
fig, ax = plt.subplots(2,1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
gca = ax[0]
gca.hist(peak_amplitude_df_fit.channel_id*5/1e3, range=(0.5*5/1e3, 10000.5*5/1e3), bins=50)
gca.set_ylabel('Counts')
gca.set_title('Number of measurements')
gca.grid()

zorder_num = 0
gca = ax[1]
for i_model, combined_channel_number in enumerate(nearby_channel_number_list):
    try:
        site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')
        # gca.plot(site_term_df.channel_id, site_term_df.site_term_P, '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
        gca.plot(site_term_df.channel_id*5/1e3, site_term_df.site_term_S + site_term_df.diff_peak_S, '-', label=f'{combined_channel_number} channels', zorder=i_model)#, Cond.# {regS.condition_number:.2f}')
    except:
        continue
    # reset the regression models
    #del regP, regS

# gca.set_yscale('log')
gca.set_ylabel('Site terms S (in log10)')
gca.set_xlabel('Channel numbers')
gca.grid()
#gca.set_xlim(10, 12)

#gca.set_xticklabels(np.arange(0, 10000, 1000))
gca.legend(loc=3, fontsize=14)
plt.savefig(regression_results_dir + '/site_terms_calibrated.png', bbox_inches='tight')
plt.savefig(regression_results_dir + '/site_terms_calibrated.pdf', bbox_inches='tight')

# %%
# Validate the strain rate for all data
weighted = ''

regression_results_dir = results_output_dir + f'/regression_results_smf_{weighted}all_coefficients_drop_4130_{min_channel}_channel_at_least'
peak_amplitude_df_all0 = pd.concat([peak_amplitude_df_fit, peak_amplitude_df_predict], axis=0)
#peak_amplitude_df_all0 = peak_amplitude_df_all0.drop(peak_amplitude_df_all0[peak_amplitude_df_all0.event_id==4130].index)

apply_secondary_calibration = False
if apply_secondary_calibration:
    calibrated_label = '_calibrated'
else:
    calibrated_label = ''

if_given_coefficents = False
nearby_channel_number_list = [100, 50, 20, 10]

    
plt.close('all')
for ii, combined_channel_number in enumerate(nearby_channel_number_list):

    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    
    peak_amplitude_df_all = combined_channels(DAS_index, peak_amplitude_df_all0, combined_channel_number)
    # only keep the channels that have been constrained
    site_term_column='combined_channel_id'
    site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df_all[site_term_column]])
    ii_channel = np.isin(site_term_keys, regS.params.keys())
    peak_amplitude_df_all = peak_amplitude_df_all[ii_channel]

    if apply_secondary_calibration:
        second_calibration = pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{combined_channel_number}chan.csv')
        peak_amplitude_df_all = pd.merge(peak_amplitude_df_all, second_calibration, on=['channel_id', 'region'])

    peak_amplitude_calculated = regS.predict(peak_amplitude_df_all)

    if if_given_coefficents:
        given_coefficients = given_coefficients_list[ii]
        mag_coef_S, dist_coef_S = given_coefficients[0], given_coefficients[1]
        peak_amplitude_calculated = peak_amplitude_calculated + (peak_amplitude_df_all.magnitude * mag_coef_S) \
                                    + np.log10(peak_amplitude_df_all.distance_in_km)*dist_coef_S
    if apply_secondary_calibration:
        peak_amplitude_calculated = peak_amplitude_calculated + peak_amplitude_df_all.diff_peak_S

    peak_amplitude_calculated = 10**peak_amplitude_calculated


    temp_peaks = np.array([np.array(peak_amplitude_df_all.peak_S), peak_amplitude_calculated]).T
    peak_comparison_df = pd.DataFrame(data=temp_peaks,
                                  columns=['peak_S', 'peak_S_predict'])

    g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S')
    g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn{calibrated_label}.png')
    g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn{calibrated_label}.pdf')


# %% 
# magnitude prediction
apply_secondary_calibration = False
if apply_secondary_calibration:
    calibrated_label = '_calibrated'
else:
    calibrated_label = ''

weighted = 'weighted_'
regression_results_dir = results_output_dir + f'/regression_results_smf_{weighted}all_coefficients_drop_4130_{min_channel}_channel_at_least'


peak_amplitude_df_all0 = pd.concat([peak_amplitude_df_fit, peak_amplitude_df_predict], axis=0)

given_coefficients_list = [None, None, None, None]

nearby_channel_number_list = [100, 50, 20, 10]
for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
    peak_amplitude_df_all = combined_channels(DAS_index, peak_amplitude_df_all0, nearby_channel_number)

    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
    
    # only keep the channels that have been constrained
    site_term_column='combined_channel_id'
    site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df_all[site_term_column]])
    ii_channel = np.isin(site_term_keys, regS.params.keys())
    peak_amplitude_df_all = peak_amplitude_df_all[ii_channel]

    # Apply to the fit set
    M_all, peak_amplitude_df_all_temp = calculate_magnitude_from_strain(peak_amplitude_df_all, regS, type='S', site_term_column=site_term_column, 
                        given_coefficients=given_coefficients_list[ii], secondary_calibration=apply_secondary_calibration)
    M_df_all = get_mean_magnitude(peak_amplitude_df_all_temp, M_all)

    gca = plot_magnitude_seaborn(M_df_all)
    if 4130 in M_df_all.index:
        gca.ax_joint.plot(M_df_all[M_df_all.index == 4130].magnitude, M_df_all[M_df_all.index == 4130].predicted_M, 'r.', markersize=15)
        gca.ax_joint.text(M_df_all[M_df_all.index == 4130].magnitude, M_df_all[M_df_all.index == 4130].predicted_M+0.2, 'clipped', color='r')

    gca.savefig(regression_results_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn{calibrated_label}.png",bbox_inches='tight')
    gca.savefig(regression_results_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn{calibrated_label}.pdf",bbox_inches='tight')
    plt.close('all')





























# # %% ==================The following is just for testing purpose============================
# # %% 
# # ==================Fit the all coefficients with partial data====================================
# # Regression OLS
# if region_label == 'sanriku':
#     nearby_channel_number_list = [100, 50, 20, 10]

#     regression_results_dir = results_output_dir + f'/regression_results_smf_all_coefficients_{min_channel}_channel_at_least_partial'
#     if not os.path.exists(regression_results_dir):
#         os.mkdir(regression_results_dir)

#     regression_parameter_txt = regression_results_dir + '/regression_slopes'
#     mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

#     for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
#         peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
#         regP, regS = fit_regression_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)

# #%% ==========================================================================================
# # Regression with weight
# if region_label == 'sanriku':
#     nearby_channel_number_list = [100, 50, 20, 10]

#     regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_all_coefficients_{min_channel}_channel_at_least_partial'
#     if not os.path.exists(regression_results_dir):
#         os.mkdir(regression_results_dir)

#     regression_parameter_txt = regression_results_dir + '/regression_slopes'
#     mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

#     for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
#         peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
#         regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)


# #%% ==========================================================================================
# # Regression without weight, drop the event 4130
# peak_amplitude_df_fit_drop = peak_amplitude_df_fit.drop(index = peak_amplitude_df_fit[peak_amplitude_df_fit.event_id == 4130].index)
# if region_label == 'sanriku':
#     nearby_channel_number_list = [100, 50, 20, 10]

#     regression_results_dir = results_output_dir + f'/regression_results_smf_all_coefficients_drop_4130_{min_channel}_channel_at_least_partial'
#     if not os.path.exists(regression_results_dir):
#         os.mkdir(regression_results_dir)

#     regression_parameter_txt = regression_results_dir + '/regression_slopes'
#     mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

#     for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
#         peak_amplitude_df_fit_drop = combined_channels(DAS_index, peak_amplitude_df_fit_drop, nearby_channel_number)
#         regP, regS = fit_regression_magnitude_range(peak_amplitude_df_fit_drop, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)

# # Regression with weight, drop the event 4130
# peak_amplitude_df_fit_drop = peak_amplitude_df_fit.drop(index = peak_amplitude_df_fit[peak_amplitude_df_fit.event_id == 4130].index)
# if region_label == 'sanriku':
#     nearby_channel_number_list = [100, 50, 20, 10]

#     regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_all_coefficients_drop_4130_{min_channel}_channel_at_least_partial'
#     if not os.path.exists(regression_results_dir):
#         os.mkdir(regression_results_dir)

#     regression_parameter_txt = regression_results_dir + '/regression_slopes'
#     mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

#     for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
#         peak_amplitude_df_fit_drop = combined_channels(DAS_index, peak_amplitude_df_fit_drop, nearby_channel_number)
#         regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df_fit_drop, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)


# # %% ==========================================================================================
# # Validate the strain rate 
# import seaborn as sns
# def plot_prediction_vs_measure_seaborn(peak_comparison_df, xy_range, phase):
#     sns.set_theme(style="ticks", font_scale=2)
#     if phase == 'P':
#         g = sns.JointGrid(data=peak_comparison_df, x="peak_P", y="peak_P_predict", marginal_ticks=True,
#                         xlim=xy_range, ylim=xy_range, height=10, space=0.3)
#     elif phase == 'S':
#         g = sns.JointGrid(data=peak_comparison_df, x="peak_S", y="peak_S_predict", marginal_ticks=True,
#                         xlim=xy_range, ylim=xy_range, height=10, space=0.3)
#     g.ax_joint.set(xscale="log")
#     g.ax_joint.set(yscale="log")

# # Create an inset legend for the histogram colorbar
#     cax = g.figure.add_axes([.65, .2, .02, .2])

# # Add the joint and marginal histogram plots 03012d
#     bins=20
#     g.plot_joint(sns.histplot, discrete=(False, False), bins=(bins, bins), cmap="light:#4D4D9C", pmax=.4, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
#     g.plot_marginals(sns.histplot, element="step", bins=bins, color="#4D4D9C")

#     g.ax_joint.plot(xy_range, xy_range, 'k-', linewidth = 2)
#     g.ax_joint.set_xlabel('measured peak strain rate\n (micro strain/s)')
#     g.ax_joint.set_ylabel('calculated peak strain rate\n (micro strain/s)')
#     return g

# # regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_all_coefficients_drop_4130_{min_channel}_channel_at_least'
# peak_amplitude_df_fit = peak_amplitude_df_fit.drop(index = peak_amplitude_df_fit[peak_amplitude_df_fit.event_id == 4130].index)
# regression_results_dir = results_output_dir + '/regression_results_smf_weighted_100_channel_at_least'


# if_given_coefficents = True


# nearby_channel_number_list = [100, 50, 20, 10]

# given_coefficients_list = [[0.591, -1.302],
#                             [0.595, -1.318],
#                             [0.608, -1.356],
#                             [0.620, -1.392]] # coefficients of weighted regression

                        
#                             # [[0.648, -1.366],
#                             # [0.653, -1.377],
#                             # [0.66, -1.396],
#                             # [0.666, -1.412]] # coefficients of unweighted regression

# plt.close('all')
# for ii, combined_channel_number in enumerate(nearby_channel_number_list):

#     given_coefficients = given_coefficients_list[ii]
#     mag_coef_S, dist_coef_S = given_coefficients[0], given_coefficients[1]

#     regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    
#     peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, combined_channel_number)
#     peak_amplitude_calculated = regS.predict(peak_amplitude_df_fit)

#     if if_given_coefficents:
#         peak_amplitude_calculated = peak_amplitude_calculated + (peak_amplitude_df_fit.magnitude * mag_coef_S) \
#                                     + np.log10(peak_amplitude_df_fit.distance_in_km)*dist_coef_S

#     peak_amplitude_calculated = 10**peak_amplitude_calculated


#     temp_peaks = np.array([np.array(peak_amplitude_df_fit.peak_S), peak_amplitude_calculated]).T
#     peak_comparison_df = pd.DataFrame(data=temp_peaks,
#                                   columns=['peak_S', 'peak_S_predict'])

#     g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S')
#     g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn.png')


# #%% ==========================================================================================
# # check those 'special' ones
# print(peak_amplitude_df_fit[peak_amplitude_calculated >1].event_id.unique())

# id_to_check = 3734
# ii = peak_amplitude_df_fit.event_id == id_to_check
# measured = peak_amplitude_df_fit[ii].peak_S
# calculated = peak_amplitude_calculated[ii]

# # 3306, 1662, 3424, 2350 are good ones, 4130 is bad???!!,
# fig, gca = plt.subplots(figsize=(8, 8))
# gca.loglog(measured, calculated, '.')
# gca.loglog([1e-2, 1e2], [1e-2, 1e2], '-k')
# gca.set_xlim(1e-2, 1e1)
# gca.set_ylim(1e-2, 1e1)

# #%% look into waveforms
# # 4130 is a special bad case that may not be S wave
# # 3734 (M4.7), 1580 (M5.4) are two large event in the prediction dataset
# id_to_check = 3734

# from scipy.signal import butter, filtfilt

# def apply_highpass_filter(event_data, f_pass, dt):
#     aa, bb = butter(4, f_pass*2*dt, 'high')
#     event_data_filter = filtfilt(aa, bb, event_data, axis=0)
#     return event_data_filter

# def show_event_data(event_data, das_time, gca, pclip=99.5):
#     if pclip is None:
#         clipVal = np.amax(abs(event_data))/20
#     else:
#         clipVal = np.percentile(np.absolute(event_data), pclip)
#     gca.imshow(event_data, 
#             extent=[0, event_data.shape[1], das_time[-1], das_time[0]],
#             aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

#     gca.set_xlabel("Channel number")
#     gca.set_ylabel("Time [s]")
#     gca.grid()
#     return gca

# event_folder = '/kuafu/EventData/Sanriku_ERI/data'
# catalog = pd.read_csv('/kuafu/EventData/Sanriku_ERI/catalog.csv')
# event_data, event_info = load_event_data(event_folder, id_to_check)
# das_time = np.arange(0, event_data.shape[0]) * event_info['dt_s']

# # Show data
# event_data = apply_highpass_filter(event_data, 0.5, event_info['dt_s'])

# fig, ax = plt.subplots(2, 1, figsize=(10,10), sharey=True)
# gca = ax[1]
# gca = show_event_data(event_data, das_time, gca, pclip=85)
# gca.set_ylim(10,40)

# gca = ax[0]
# gca.plot(event_data[:, ::1000] + np.cumsum(1*np.ones((event_data[:, ::1000].shape[1]))), das_time, linewidth=0.5)
# gca.set_xticklabels('')
# gca.set_ylim(10,40)

# plt.savefig(results_output_dir + f'/event_{id_to_check}_waveforms.png', bbox_inches='tight')