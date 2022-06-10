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

def split_P_S_dataframe(peak_amplitude_df):
    # use P and S separately to do the regression
    peak_amplitude_df_P = peak_amplitude_df[['peak_P', 'magnitude', 'distance_in_km', 'combined_channel_id']]
    peak_amplitude_df_S = peak_amplitude_df[['peak_S', 'magnitude', 'distance_in_km', 'combined_channel_id']]

    # Remove some extreme data outliers before fitting
    peak_amplitude_df_P = peak_amplitude_df_P.dropna()
    peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.peak_P>0]
    peak_amplitude_df_P = peak_amplitude_df_P.drop(peak_amplitude_df_P[(peak_amplitude_df_P.peak_P > 1e3)].index)

    peak_amplitude_df_S = peak_amplitude_df_S.dropna()
    peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.peak_S>0]
    peak_amplitude_df_S = peak_amplitude_df_S.drop(peak_amplitude_df_S[(peak_amplitude_df_S.peak_S > 1e3)].index)

    return peak_amplitude_df_P, peak_amplitude_df_S

def fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None):
   
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)

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


def fit_regression_with_weight_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None):
    """ Linear regression with weight, the weight is 10**magnitude """

    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)
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

def calculate_magnitude_from_strain(peak_amplitude_df, reg, type, site_term_column='region_site', given_coefficients=None):

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

    elif type == 'S':
        M_predict = (np.log10(peak_amplitude_df.peak_S) \
                    - np.array(reg.params[site_term_keys]) \
                    - dist_coef * np.log10(peak_amplitude_df.distance_in_km)) \
                    / mag_coef

    return M_predict

def get_mean_magnitude(peak_amplitude_df, M_predict):
    temp_df = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df['predicted_M'] = M_predict
    temp_df = temp_df.groupby(temp_df['event_id']).aggregate('mean')

    temp_df2 = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df2['predicted_M_std'] = M_predict
    temp_df2 = temp_df2.groupby(temp_df2['event_id']).aggregate(np.std)

    temp_df = pd.concat([temp_df, temp_df2['predicted_M_std']], axis=1)
    return temp_df

# ==============================  Ridgecrest data ========================================
#%% Specify the file names
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events'
das_pick_file_name = 'peak_amplitude.csv'
region_label = 'ridgecrest'

# ==============================  Mammoth data - South========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events'
das_pick_file_name = 'peak_amplitude.csv'
region_label = 'mammothS'

# ==============================  Mammoth data - North========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events'
das_pick_file_name = 'peak_amplitude.csv'
region_label = 'mammothN'

# ==============================  Sanriku data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events'
das_pick_file_name = 'peak_amplitude.csv'
region_label = 'sanriku'

#%% load the peak amplitude results
# Load the peak amplitude results for regression
snr_threshold = 10
M_threshold = [2, 10]
peak_amplitude_df_fit, DAS_index = load_and_add_region(das_pick_file_folder + '/' + das_pick_file_name, 
                                                   region_label=region_label, snr_threshold=snr_threshold)
peak_amplitude_df_fit = peak_amplitude_df_fit[(peak_amplitude_df_fit.magnitude >= M_threshold[0]) & 
                                              (peak_amplitude_df_fit.magnitude <= M_threshold[1]) &
                                              (peak_amplitude_df_fit.QA == 'Yes')]

# Preprocessing the peak amplitude data
#peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df_fit = add_event_label(peak_amplitude_df_fit)

fit_data_info_str = f'================ {len(peak_amplitude_df_fit.event_id.unique())} events used to constrain site terms ============\n' + \
                    f'SNR >= {snr_threshold} \n' + \
                    f'Magnitude: {M_threshold} \n' + \
                    f'{peak_amplitude_df_fit.shape[0]} measurements \n\n'
print(fit_data_info_str)

#%% ==========================================================================================
# Fit the site term only
# Regression OLS
if region_label == 'sanriku':
    given_coefficients_list = [[np.nan, np.nan, 0.618, -1.221],
                               [np.nan, np.nan, 0.623, -1.207],
                               [np.nan, np.nan, 0.630, -1.192],
                               [np.nan, np.nan, 0.635, -1.182]]

    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + '/regression_results_smf'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
        given_coefficients = given_coefficients_list[ii]
        regP, regS = fit_regression_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=given_coefficients)

#%% ==========================================================================================
# Regression with weight
if region_label == 'sanriku':
    given_coefficients_list = [[np.nan, np.nan, 0.586, -1.203],
                               [np.nan, np.nan, 0.59, -1.218],
                               [np.nan, np.nan, 0.602, -1.25],
                               [np.nan, np.nan, 0.613, -1.281]]

    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + '/regression_results_smf_weighted'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
        given_coefficients = given_coefficients_list[ii]
        regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=given_coefficients)


#%% ==========================================================================================
# Fit the all coefficients
# Regression OLS
if region_label == 'sanriku':
    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + '/regression_results_smf_all_coefficients'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
        regP, regS = fit_regression_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)

#%% ==========================================================================================
# Regression with weight
if region_label == 'sanriku':
    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + '/regression_results_smf_weighted_all_coefficients'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
        regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)


#%% ==========================================================================================
# Regression without weight, drop the event 4130
peak_amplitude_df_fit_drop = peak_amplitude_df_fit.drop(index = peak_amplitude_df_fit[peak_amplitude_df_fit.event_id == 4130].index)
if region_label == 'sanriku':
    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + '/regression_results_smf_all_coefficients_drop_4130'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_fit_drop = combined_channels(DAS_index, peak_amplitude_df_fit_drop, nearby_channel_number)
        regP, regS = fit_regression_magnitude_range(peak_amplitude_df_fit_drop, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)

# Regression with weight, drop the event 4130
peak_amplitude_df_fit_drop = peak_amplitude_df_fit.drop(index = peak_amplitude_df_fit[peak_amplitude_df_fit.event_id == 4130].index)
if region_label == 'sanriku':
    nearby_channel_number_list = [100, 50, 20, 10]

    regression_results_dir = results_output_dir + '/regression_results_smf_weighted_all_coefficients_drop_4130'
    if not os.path.exists(regression_results_dir):
        os.mkdir(regression_results_dir)

    regression_parameter_txt = regression_results_dir + '/regression_slopes'
    mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

    for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
        peak_amplitude_df_fit_drop = combined_channels(DAS_index, peak_amplitude_df_fit_drop, nearby_channel_number)
        regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df_fit_drop, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None)


# %% ==========================================================================================
# Validate the strain rate 
import seaborn as sns
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
    g.plot_joint(sns.histplot, discrete=(False, False), cmap="light:#4D4D9C", pmax=.2, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", color="#4D4D9C")

    g.ax_joint.plot(xy_range, xy_range, 'k-', linewidth = 2)
    g.ax_joint.set_xlabel('measured peak')
    g.ax_joint.set_ylabel('calculated peak')
    return g

regression_results_dir = results_output_dir + '/regression_results_smf_all_coefficients_drop_4130'
peak_amplitude_df_fit = peak_amplitude_df_fit.drop(index = peak_amplitude_df_fit[peak_amplitude_df_fit.event_id == 4130].index)
# regression_results_dir = results_output_dir + '/regression_results_smf_weighted_all_coefficients'


if_given_coefficents = False


nearby_channel_number_list = [100, 50, 20, 10]

given_coefficients_list = [[0.586, -1.203],
                           [0.59, -1.218],
                           [0.602, -1.25],
                           [0.613, -1.281]]

plt.close('all')
for ii, combined_channel_number in enumerate(nearby_channel_number_list):

    given_coefficients = given_coefficients_list[ii]
    mag_coef_S, dist_coef_S = given_coefficients[0], given_coefficients[1]

    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    
    peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, combined_channel_number)
    peak_amplitude_calculated = regS.predict(peak_amplitude_df_fit)

    if if_given_coefficents:
        peak_amplitude_calculated = peak_amplitude_calculated + (peak_amplitude_df_fit.magnitude * mag_coef_S) \
                                    + np.log10(peak_amplitude_df_fit.distance_in_km)*dist_coef_S

    peak_amplitude_calculated = 10**peak_amplitude_calculated


    temp_peaks = np.array([np.array(peak_amplitude_df_fit.peak_S), peak_amplitude_calculated]).T
    peak_comparison_df = pd.DataFrame(data=temp_peaks,
                                  columns=['peak_S', 'peak_S_predict'])

    g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S')
    g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_seaborn.png')



#%% ==========================================================================================
# check those 'special' ones
print(peak_amplitude_df_fit[peak_amplitude_calculated >1].event_id.unique())

id_to_check = 3734
ii = peak_amplitude_df_fit.event_id == id_to_check
measured = peak_amplitude_df_fit[ii].peak_S
calculated = peak_amplitude_calculated[ii]

# 3306, 1662, 3424, 2350 are good ones, 4130 is bad???!!,
fig, gca = plt.subplots(figsize=(8, 8))
gca.loglog(measured, calculated, '.')
gca.loglog([1e-2, 1e2], [1e-2, 1e2], '-k')
gca.set_xlim(1e-2, 1e1)
gca.set_ylim(1e-2, 1e1)

#%% look into waveforms
# 4130 is a special bad case that may not be S wave
# 3734 (M4.7), 1580 (M5.4) are two large event in the prediction dataset
id_to_check = 3734

from scipy.signal import butter, filtfilt

def apply_highpass_filter(event_data, f_pass, dt):
    aa, bb = butter(4, f_pass*2*dt, 'high')
    event_data_filter = filtfilt(aa, bb, event_data, axis=0)
    return event_data_filter

def show_event_data(event_data, das_time, gca, pclip=99.5):
    if pclip is None:
        clipVal = np.amax(abs(event_data))/20
    else:
        clipVal = np.percentile(np.absolute(event_data), pclip)
    gca.imshow(event_data, 
            extent=[0, event_data.shape[1], das_time[-1], das_time[0]],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

    gca.set_xlabel("Channel number")
    gca.set_ylabel("Time [s]")
    gca.grid()
    return gca

event_folder = '/kuafu/EventData/Sanriku_ERI/data'
catalog = pd.read_csv('/kuafu/EventData/Sanriku_ERI/catalog.csv')
event_data, event_info = load_event_data(event_folder, id_to_check)
das_time = np.arange(0, event_data.shape[0]) * event_info['dt_s']

# Show data
event_data = apply_highpass_filter(event_data, 0.5, event_info['dt_s'])

fig, ax = plt.subplots(2, 1, figsize=(10,10), sharey=True)
gca = ax[1]
gca = show_event_data(event_data, das_time, gca, pclip=85)
gca.set_ylim(10,40)

gca = ax[0]
gca.plot(event_data[:, ::1000] + np.cumsum(1*np.ones((event_data[:, ::1000].shape[1]))), das_time, linewidth=0.5)
gca.set_xticklabels('')
gca.set_ylim(10,40)

plt.savefig(results_output_dir + f'/event_{id_to_check}_waveforms.png', bbox_inches='tight')

# %% ==========================================================================================
# Check site terms
# regression_results_dir = results_output_dir + '/regression_results_smf_weighted'
regression_results_dir = results_output_dir + '/regression_results_smf'

plt.close('all')
fig, ax = plt.subplots(2,1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
gca = ax[0]
gca.hist(peak_amplitude_df_fit.channel_id, range=(0.5, 10000.5), bins=1000)
gca.set_ylabel('Counts')
gca.set_title('Number of measurements')
gca.grid()

zorder_num = 0
gca = ax[1]
for combined_channel_number in nearby_channel_number_list:
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")

    peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, combined_channel_number)
    combined_channel_id = np.sort(peak_amplitude_df_fit.combined_channel_id.unique())

    # Compare all the site terms
    site_term_S = regS.params

    zorder_num -= 1
    gca.plot(combined_channel_id * combined_channel_number, 10**site_term_S, '-', 
             label=f'{combined_channel_number} channels', zorder=zorder_num)

gca.set_yscale('log')
gca.set_ylabel('Site terms from S wave')
gca.set_xlabel('Channel numbers')
gca.grid()

#gca.set_xticklabels(np.arange(0, 10000, 1000))
gca.legend(loc=1, fontsize=14)
plt.savefig(regression_results_dir + '/site_terms.png', bbox_inches='tight')


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
    g.plot_marginals(sns.histplot, element="step", color="#9C4D4D")

    g.ax_joint.plot([0,8], [0,8], 'k-', linewidth = 2)
    g.ax_joint.plot([0,8], [1,9], 'k--', linewidth = 1)
    g.ax_joint.plot([0,8], [-1,7], 'k--', linewidth = 1)
    g.ax_joint.set_xlabel('catalog magnitude')
    g.ax_joint.set_ylabel('predicted magnitude')

    return g

# %% ==========================================================================================
# Use the site term to estimate magnitude
# Load the peak amplitude results
snr_threshold = 5
M_threshold = [1, 10]

peak_amplitude_df_predict, DAS_index = load_and_add_region(das_pick_file_folder + '/' + das_pick_file_name, 
                                                   region_label=region_label, snr_threshold=snr_threshold)

# Preprocessing the peak amplitude data
#peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df_predict = add_event_label(peak_amplitude_df_predict)

peak_amplitude_df_predict = peak_amplitude_df_predict.drop(index=peak_amplitude_df_fit.index)
      
peak_amplitude_df_predict = peak_amplitude_df_predict[(peak_amplitude_df_predict.magnitude >= M_threshold[0]) & 
                                                      (peak_amplitude_df_predict.magnitude <= M_threshold[1]) & 
                                                      (peak_amplitude_df_predict.QA == 'Yes')]
predict_data_info_str = f'================ {len(peak_amplitude_df_predict.event_id.unique())} events for validation =============\n' + \
                        f'SNR >= {snr_threshold} \n' + \
                        f'Magnitude: {M_threshold} \n' + \
                        f'{peak_amplitude_df_predict.shape[0]} measurements \n\n'
print(predict_data_info_str)

#%% ==========================================================================================
# do the magnitude estimation with the regression results
regression_results_dir = results_output_dir + '/regression_results_smf_weighted'
# regression_results_dir = results_output_dir + '/regression_results_smf_weighted'

given_coefficients_list = [[0.586, -1.203],
                           [0.59, -1.218],
                           [0.602, -1.25],
                           [0.613, -1.281]]
# given_coefficients_list = [None, None, None, None]

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
    M_fit = calculate_magnitude_from_strain(peak_amplitude_df_fit, regS, type='S', site_term_column=site_term_column, given_coefficients=given_coefficients_list[ii])
    M_df_fit = get_mean_magnitude(peak_amplitude_df_fit, M_fit)

    # Apply to the prediction set
    M_predict = calculate_magnitude_from_strain(peak_amplitude_df_predict, regS, type='S', site_term_column=site_term_column, given_coefficients=given_coefficients_list[ii])
    M_df_predict = get_mean_magnitude(peak_amplitude_df_predict, M_predict)

    gca = plot_magnitude_seaborn(M_df_predict)
    gca.ax_joint.plot(M_df_fit.magnitude, M_df_fit.predicted_M, 'k.', markersize=10)
    gca.savefig(regression_results_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
    plt.close('all')

#%% ==========================================================================================
# write some data info
with open(results_output_dir + '/data_info.txt', 'w') as f:
    f.write(fit_data_info_str)
    f.write(predict_data_info_str)
# %%
