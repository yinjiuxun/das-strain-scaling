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

        regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
        regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()

        print(regP.params[-2:])
        print(regS.params[-2:])
        print('\n\n')
        write_regression_summary(regression_results_dir, file_name_P, regP)
        write_regression_summary(regression_results_dir, file_name_S, regS)
        regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
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
        regP = smf.wls(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', 
                    data=peak_amplitude_df_P, weights = (10**peak_amplitude_df_P.magnitude)).fit()
        regS = smf.wls(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', 
                    data=peak_amplitude_df_S, weights = (10**peak_amplitude_df_S.magnitude)).fit()

        print(regP.params[-2:])
        print(regS.params[-2:])
        print('\n\n')
        write_regression_summary(regression_results_dir, file_name_P, regP)
        write_regression_summary(regression_results_dir, file_name_S, regS)
        regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
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
# Load the peak amplitude results
snr_threshold = 10
peak_amplitude_df, DAS_index = load_and_add_region(das_pick_file_folder + '/' + das_pick_file_name, 
                                                   region_label=region_label, snr_threshold=snr_threshold)

# Preprocessing the peak amplitude data
#peak_amplitude_df = peak_amplitude_df.dropna()
peak_amplitude_df = add_event_label(peak_amplitude_df)

#%% 
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
        peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
        given_coefficients = given_coefficients_list[ii]

        M_threshold = [2, 10]
        regP, regS = fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=given_coefficients)
        
        # reset the regression models
        #del regP, regS

#%% 
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
        peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
        given_coefficients = given_coefficients_list[ii]

        M_threshold = [2, 10]
        regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=given_coefficients)
        
        # reset the regression models
        #del regP, regS


# %%
import seaborn as sns
def plot_magnitude_seaborn(df_magnitude):
    sns.set_theme(style="ticks", font_scale=2)

    g = sns.JointGrid(data=df_magnitude, x="magnitude", y="predicted_M", marginal_ticks=True,
    xlim=(1, 7), ylim=(1, 7), height=10, space=0.3)


    # Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

    # Add the joint and marginal histogram plots 03012d
    g.plot_joint(
    sns.histplot, discrete=(False, False),
    cmap="light:#9C4D4D", pmax=.7, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", color="#9C4D4D")

    g.ax_joint.plot([1,8], [1,8], 'k-', linewidth = 2)
    g.ax_joint.plot([1,8], [2,9], 'k--', linewidth = 1)
    g.ax_joint.plot([1,8], [0,7], 'k--', linewidth = 1)
    g.ax_joint.set_xlabel('catalog magnitude')
    g.ax_joint.set_ylabel('predicted magnitude')

    return g

# %% 
# Use the site term to estimate magnitude
M_threshold = [2, 10]
peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
nearby_channel_number_list = [100, 50, 20, 10]

#regression_results_dir = results_output_dir + '/regression_results_smf'
regression_results_dir = results_output_dir + '/regression_results_smf_weighted'

given_coefficients_list = [[0.586, -1.203],
                           [0.59, -1.218],
                           [0.602, -1.25],
                           [0.613, -1.281]]

for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number)
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")

    M_predict = calculate_magnitude_from_strain(peak_amplitude_df, regS, type='S', site_term_column='combined_channel_id', given_coefficients=given_coefficients_list[ii])
    temp_df = get_mean_magnitude(peak_amplitude_df, M_predict)

    gca = plot_magnitude_seaborn(temp_df)
    gca.savefig(regression_results_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
    plt.close('all')


# %%
