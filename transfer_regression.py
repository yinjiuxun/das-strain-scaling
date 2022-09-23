#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import shutil
import statsmodels.api as sm
import random

from utility.general import mkdir
from utility.processing import filter_by_channel_number, split_P_S_dataframe, filter_event
from utility.regression import fit_regression_transfer

#%% 
# define some local function for convenience
def split_fit_and_predict(N_event_fit, peak_amplitude_df):
    """Randomly choose a few events to fit"""
    event_id_all =  peak_amplitude_df.event_id.unique()
    event_id_fit = random.choices(event_id_all, k=N_event_fit) # event id list for regression fit
    event_id_predict = list(set(event_id_all) - set(event_id_fit)) # event id lsit for validation 

    peak_amplitude_df_fit = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit)]
    peak_amplitude_df_predict = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_predict)]

    return event_id_fit, peak_amplitude_df_fit, event_id_predict, peak_amplitude_df_predict

def transfer_fitting(regP_pre, regS_pre, peak_amplitude_df_fit, weighted):
    """Transfer scaling to obtain site terms"""
    site_term_P = fit_regression_transfer(peak_amplitude_df_fit, regP_pre, wavetype='P', weighted=weighted, M_threshold=M_threshold, snr_threshold=snr_threshold_transfer, min_channel=min_channel)
    site_term_S = fit_regression_transfer(peak_amplitude_df_fit, regS_pre, wavetype='S', weighted=weighted, M_threshold=M_threshold, snr_threshold=snr_threshold_transfer, min_channel=min_channel)

    # combine P and S
    site_term_df = pd.merge(site_term_P, site_term_S, on='channel_id', how='outer')
    site_term_df['region'] = region_text
    site_term_df = site_term_df.iloc[:, [0, 3, 1, 2]]
    return site_term_df

#%%
# some parameters
snr_threshold_transfer = 5
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [0, 10]

weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise
#%%
# Coefficients from previous results
previous_regression_dir = f'/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
regP_pre_path = previous_regression_dir + f"/P_regression_combined_site_terms_iter.pickle"
regS_pre_path = previous_regression_dir + f"/S_regression_combined_site_terms_iter.pickle"

regP_pre = sm.load(regP_pre_path)
regS_pre = sm.load(regS_pre_path)



# #%%
# # load the data
# results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
# region_text = 'LA-Google'
# peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
# peak_amplitude_df = filter_event(peak_amplitude_df, snr_threshold=snr_threshold_transfer, min_channel=min_channel)

# #%%
# # Randomly choose a few events to fit
# # number of event for transfer fitting
# N_event_fit = 10
# random.seed(111)
# event_id_fit, peak_amplitude_df_fit, event_id_predict, peak_amplitude_df_predict = split_fit_and_predict(N_event_fit, peak_amplitude_df)

# #%%
# # Transfer scaling to obtain site terms
# site_term_df = transfer_fitting(regP_pre, regS_pre, peak_amplitude_df_fit, weighted)

# #%%
# # make output directory and output results
# results_output_dir = results_output_dir
# regression_results_dir = results_output_dir + f'/transfer_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
# mkdir(regression_results_dir)
# site_term_df.to_csv(regression_results_dir + '/site_terms_transfer.csv', index=False)

# # output the event id list of fit and predict events
# np.savez(regression_results_dir + '/transfer_event_list.npz', event_id_fit=event_id_fit, event_id_predict=event_id_predict)

# # also copy the regression results to the results directory
# shutil.copyfile(regP_pre_path, regression_results_dir + '/P_regression_combined_site_terms_transfer.pickle')
# shutil.copyfile(regS_pre_path, regression_results_dir + '/S_regression_combined_site_terms_transfer.pickle')

#%%
# TODO: systematically test the fittig events
results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
region_text = 'LA-Google'
peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
peak_amplitude_df = filter_event(peak_amplitude_df, snr_threshold=snr_threshold_transfer, min_channel=min_channel)
event_id_all =  peak_amplitude_df.event_id.unique()

random.seed(212)
N_event_fit_list = range(2, 30)
N_test = 50

for N_event_fit in N_event_fit_list:
    for i_test in range(N_test):
        print(f'================= {N_event_fit} to fit, {i_test}th test =====================')
        # Randomly choose a few events to fit
        event_id_fit, peak_amplitude_df_fit, event_id_predict, peak_amplitude_df_predict = split_fit_and_predict(N_event_fit, peak_amplitude_df)
        # Transfer scaling to obtain site terms
        site_term_df = transfer_fitting(regP_pre, regS_pre, peak_amplitude_df_fit, weighted)

        # make output directory and output results
        results_output_dir = results_output_dir
        regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least'
        mkdir(regression_results_dir)
        
        regression_results_dir = regression_results_dir + f'/{N_event_fit}_fit_events_{i_test}th_test'
        mkdir(regression_results_dir)

        site_term_df.to_csv(regression_results_dir + '/site_terms_transfer.csv', index=False)

        # output the event id list of fit and predict events
        np.savez(regression_results_dir + '/transfer_event_list.npz', event_id_fit=event_id_fit, event_id_predict=event_id_predict)

        # also copy the regression results to the results directory
        shutil.copyfile(regP_pre_path, regression_results_dir + '/P_regression_combined_site_terms_transfer.pickle')
        shutil.copyfile(regS_pre_path, regression_results_dir + '/S_regression_combined_site_terms_transfer.pickle')





# %%TODO: applied to Sanriku as well.


















#### WILL BE REMOVED SOON ###
#%%
# def write_regression_summary(regression_results_dir, file_name, reg):
#     # make a directory to store the regression results in text
#     regression_text = regression_results_dir + '/regression_results_txt'
#     if not os.path.exists(regression_text):
#         os.mkdir(regression_text)
#     with open(regression_text + '/' + file_name + '.txt', "w") as text_file:
#         text_file.write(reg.summary().as_text())

# def filter_by_channel_number(peak_amplitude_df, min_channel):
#     """To remove the measurements from few channels (< min_channel)"""
#     event_channel_count = peak_amplitude_df.groupby(['event_id'])['event_id'].count()
#     channel_count = event_channel_count.values
#     event_id = event_channel_count.index
#     event_id = event_id[channel_count >= min_channel]

#     return peak_amplitude_df[peak_amplitude_df['event_id'].isin(event_id)]

# def split_P_S_dataframe(peak_amplitude_df):
#     # use P and S separately to do the regression
#     peak_amplitude_df_P = peak_amplitude_df[['event_id', 'peak_P', 'magnitude', 'distance_in_km', 'combined_channel_id']]
#     peak_amplitude_df_S = peak_amplitude_df[['event_id', 'peak_S', 'magnitude', 'distance_in_km', 'combined_channel_id']]

#     # Remove some extreme data outliers before fitting
#     peak_amplitude_df_P = peak_amplitude_df_P.dropna()
#     peak_amplitude_df_P = peak_amplitude_df_P[peak_amplitude_df_P.peak_P>0]
#     peak_amplitude_df_P = peak_amplitude_df_P.drop(peak_amplitude_df_P[(peak_amplitude_df_P.peak_P > 1e3)].index)

#     peak_amplitude_df_S = peak_amplitude_df_S.dropna()
#     peak_amplitude_df_S = peak_amplitude_df_S[peak_amplitude_df_S.peak_S>0]
#     peak_amplitude_df_S = peak_amplitude_df_S.drop(peak_amplitude_df_S[(peak_amplitude_df_S.peak_S > 1e3)].index)

#     return peak_amplitude_df_P, peak_amplitude_df_S

# def fit_regression_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None, min_channel=None):
   
#     peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
#     # use P and S separately to do the regression
#     peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)
    
#     # Before regression, first filter events based on the channel number, if < 100, discard the event (probably a wrong pick/local noise)
#     if min_channel:
#         peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
#         peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

#     file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
#     file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"
#     regP, regS = None, None

#     if given_coefficients is None:
#         print('Fit for all coefficients........\n')
#         if peak_amplitude_df_P.shape[0] != 0:
#             regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
#             print(regP.params[-2:])
#             write_regression_summary(regression_results_dir, file_name_P, regP)
#             regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)

#         if peak_amplitude_df_S.shape[0] != 0:
#             regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()      
#             print(regS.params[-2:])
#             write_regression_summary(regression_results_dir, file_name_S, regS)
#             regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
        
#     else: # given the coefficients from other array, fit for the site terms
#         if (~np.isnan(given_coefficients[0])) & (~np.isnan(given_coefficients[1])):
#             mag_coef_P, dist_coef_P = given_coefficients[0], given_coefficients[1]
#             peak_amplitude_df_P.peak_P = peak_amplitude_df_P.peak_P / 10**(peak_amplitude_df_P.magnitude * mag_coef_P) / peak_amplitude_df_P.distance_in_km**dist_coef_P
            
#             regP = smf.ols(formula='np.log10(peak_P) ~ C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
#             regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
#             #write_regression_summary(regression_results_dir, file_name_P, regP)

#         if (~np.isnan(given_coefficients[2])) & (~np.isnan(given_coefficients[3])):
#             mag_coef_S, dist_coef_S = given_coefficients[2], given_coefficients[3]
#             peak_amplitude_df_S.peak_S = peak_amplitude_df_S.peak_S / 10**(peak_amplitude_df_S.magnitude * mag_coef_S) / peak_amplitude_df_S.distance_in_km**dist_coef_S
            
#             regS = smf.ols(formula='np.log10(peak_S) ~ C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()
#             regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
#             #write_regression_summary(regression_results_dir, file_name_S, regS)
    
#     return regP,regS


# def fit_regression_with_weight_magnitude_range(peak_amplitude_df, M_threshold, regression_results_dir, nearby_channel_number, given_coefficients=None, min_channel=None):
#     """ Linear regression with weight, the weight is 10**magnitude """

#     peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
#     # use P and S separately to do the regression
#     peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)

#     # Before regression, first filter events based on the channel number, if < 100, discard the event (probably a wrong pick/local noise)
#     if min_channel:
#         peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
#         peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

#     file_name_P = f"/P_regression_combined_site_terms_{nearby_channel_number}chan"
#     file_name_S = f"/S_regression_combined_site_terms_{nearby_channel_number}chan"
#     regP, regS = None, None

#     if given_coefficients is None:
#         print('Fit for all coefficients........\n')
#         if peak_amplitude_df_P.shape[0] != 0:
#             regP = smf.wls(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', 
#                         data=peak_amplitude_df_P, weights = (10**peak_amplitude_df_P.magnitude)).fit()
#             print(regP.params[-2:])
#             write_regression_summary(regression_results_dir, file_name_P, regP)
#             regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)

#         if peak_amplitude_df_S.shape[0] != 0:
#             regS = smf.wls(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(combined_channel_id) - 1', 
#                         data=peak_amplitude_df_S, weights = (10**peak_amplitude_df_S.magnitude)).fit()
#             print(regS.params[-2:])
#             write_regression_summary(regression_results_dir, file_name_S, regS)
#             regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)

#     else: # given the coefficients from other array, fit for the site terms
#         if (~np.isnan(given_coefficients[0])) & (~np.isnan(given_coefficients[1])):
#             mag_coef_P, dist_coef_P = given_coefficients[0], given_coefficients[1]
#             peak_amplitude_df_P.peak_P = peak_amplitude_df_P.peak_P / 10**(peak_amplitude_df_P.magnitude * mag_coef_P) / peak_amplitude_df_P.distance_in_km**dist_coef_P
            
#             regP = smf.ols(formula='np.log10(peak_P) ~ C(combined_channel_id) - 1', data=peak_amplitude_df_P).fit()
#             regP.save(regression_results_dir + '/' + file_name_P + '.pickle', remove_data=True)
#             #write_regression_summary(regression_results_dir, file_name_P, regP)

#         if (~np.isnan(given_coefficients[2])) & (~np.isnan(given_coefficients[3])):
#             mag_coef_S, dist_coef_S = given_coefficients[2], given_coefficients[3]
#             peak_amplitude_df_S.peak_S = peak_amplitude_df_S.peak_S / 10**(peak_amplitude_df_S.magnitude * mag_coef_S) / peak_amplitude_df_S.distance_in_km**dist_coef_S
            
#             regS = smf.ols(formula='np.log10(peak_S) ~ C(combined_channel_id) - 1', data=peak_amplitude_df_S).fit()
#             regS.save(regression_results_dir + '/' + file_name_S + '.pickle', remove_data=True)
#             #write_regression_summary(regression_results_dir, file_name_S, regS)
            
#     return regP,regS

# def secondary_site_calibration(regP, regS, peak_amplitude_df, given_coefficients=None):
#     """obtain secondary site term calibration"""

#     peak_amplitude_df_temp = pd.DataFrame(columns=['region_site', 'channel_id', 'region', 'magnitude', 'diff_peak_P', 'diff_peak_S', 'site_calibate_P', 'site_calibrate_S'])
#     peak_amplitude_df_temp.channel_id = peak_amplitude_df.channel_id
#     peak_amplitude_df_temp.region = peak_amplitude_df.region
#     peak_amplitude_df_temp.magnitude = peak_amplitude_df.magnitude

#     if regP is None:
#         peak_amplitude_df_temp.diff_peak_P = np.nan
#     else:
#         y_P_predict = regP.predict(peak_amplitude_df)
#         if given_coefficients:
#             mag_coef_P, dist_coef_P = given_coefficients[0], given_coefficients[1]
#             y_P_predict = y_P_predict + (peak_amplitude_df.magnitude * mag_coef_P) + dist_coef_P * np.log10(peak_amplitude_df.distance_in_km)

#         peak_amplitude_df_temp.diff_peak_P = (-y_P_predict + np.log10(peak_amplitude_df.peak_P))#*10**peak_amplitude_df.magnitude/weighted_all
    
#     if regS is None:
#         peak_amplitude_df_temp.diff_peak_S = np.nan
#     else:
#         y_S_predict = regS.predict(peak_amplitude_df)
#         if given_coefficients:
#             mag_coef_S, dist_coef_S = given_coefficients[2], given_coefficients[3]
#             y_S_predict = y_S_predict + peak_amplitude_df.magnitude * mag_coef_S + dist_coef_S * np.log10(peak_amplitude_df.distance_in_km)
        
#         peak_amplitude_df_temp.diff_peak_S = (-y_S_predict + np.log10(peak_amplitude_df.peak_S))#*10**peak_amplitude_df.magnitude/weighted_all
        
#     peak_amplitude_df_temp.region_site = peak_amplitude_df_temp.region + '-' + peak_amplitude_df.combined_channel_id.astype('str')

#     second_calibration = peak_amplitude_df_temp.groupby(['channel_id', 'region'], as_index=False).mean()
#     temp_df = peak_amplitude_df_temp[['region_site', 'channel_id', 'region']].drop_duplicates(subset=['channel_id', 'region_site'])
#     second_calibration = pd.merge(second_calibration, temp_df, on=['channel_id', 'region'])
#     second_calibration = second_calibration.drop(columns=['magnitude'])
#     return second_calibration

# def calculate_magnitude_from_strain(peak_amplitude_df, reg, type, site_term_column='region_site', given_coefficients=None, secondary_calibration=None):
    
#     if secondary_calibration:
#         second_calibration = pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv')
#         peak_amplitude_df = pd.merge(peak_amplitude_df, second_calibration, on=['channel_id', 'region'])
#         # MUST BE VERY CAREFUL WHEN USING merge, it changes the order of DataFrame!

#     # get the annoying categorical keys
#     try:
#         site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
#     except:
#         raise NameError(f"Index {site_term_column} doesn't exist!")

#     if given_coefficients is None:
#         mag_coef, dist_coef = reg.params['magnitude'], reg.params['np.log10(distance_in_km)']
#     else:
#         mag_coef, dist_coef = given_coefficients[0], given_coefficients[1]

#     if type == 'P':
#         M_predict = (np.log10(peak_amplitude_df.peak_P) \
#                     - np.array(reg.params[site_term_keys]) \
#                     - dist_coef * np.log10(peak_amplitude_df.distance_in_km)) \
#                     / mag_coef
#         if secondary_calibration:
#                 M_predict = M_predict.copy() - peak_amplitude_df.diff_peak_P/mag_coef

#     elif type == 'S':
#         M_predict = (np.log10(peak_amplitude_df.peak_S) \
#                     - np.array(reg.params[site_term_keys]) \
#                     - dist_coef * np.log10(peak_amplitude_df.distance_in_km)) \
#                     / mag_coef
#         if secondary_calibration:
#                 M_predict = M_predict.copy() - peak_amplitude_df.diff_peak_S/mag_coef

#     return M_predict, peak_amplitude_df

# def get_mean_magnitude(peak_amplitude_df, M_predict):
#     temp_df = peak_amplitude_df[['event_id', 'magnitude']].copy()
#     temp_df['predicted_M'] = M_predict
#     temp_df = temp_df.groupby(temp_df['event_id']).aggregate('mean')

#     temp_df2 = peak_amplitude_df[['event_id', 'magnitude']].copy()
#     temp_df2['predicted_M_std'] = M_predict
#     temp_df2 = temp_df2.groupby(temp_df2['event_id']).aggregate(np.std)

#     temp_df = pd.concat([temp_df, temp_df2['predicted_M_std']], axis=1)
#     return temp_df

# # %% ==========================================================================================
# # Apply to more data to predict the magnitude
# import seaborn as sns
# def plot_magnitude_seaborn(df_magnitude):
#     sns.set_theme(style="ticks", font_scale=2)

#     g = sns.JointGrid(data=df_magnitude, x="magnitude", y="predicted_M", marginal_ticks=True,
#     xlim=(0, 7), ylim=(0, 7), height=10, space=0.3)


#     # Create an inset legend for the histogram colorbar
#     cax = g.figure.add_axes([.65, .2, .02, .2])

#     # Add the joint and marginal histogram plots 03012d
#     g.plot_joint(
#     sns.histplot, discrete=(False, False),
#     cmap="light:#9C4D4D", pmax=.7, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
#     g.plot_marginals(sns.histplot, element="step", color="#9C4D4D") #flare; light:#9C4D4D; dark:salmon_r

#     g.ax_joint.plot([0,8], [0,8], 'k-', linewidth = 2)
#     g.ax_joint.plot([0,8], [1,9], 'k--', linewidth = 1)
#     g.ax_joint.plot([0,8], [-1,7], 'k--', linewidth = 1)
#     g.ax_joint.set_xlabel('catalog magnitude')
#     g.ax_joint.set_ylabel('predicted magnitude')

#     return g

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

#     # Create an inset legend for the histogram colorbar
#     cax = g.figure.add_axes([.65, .2, .02, .2])

#     # Add the joint and marginal histogram plots 03012d
#     bins=40
#     g.plot_joint(sns.histplot, discrete=(False, False), bins=(bins, bins), cmap="light:#4D4D9C", pmax=.6, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
#     g.plot_marginals(sns.histplot, element="step", bins=bins, binrange=(-2, 2), color="#4D4D9C")

#     g.ax_joint.plot(xy_range, xy_range, 'k-', linewidth = 2)
#     g.ax_joint.set_xlabel('measured peak strain rate\n (micro strain/s)')
#     g.ax_joint.set_ylabel('calculated peak strain rate\n (micro strain/s)')
#     return g

# # ==============================  LAX data ========================================
# #%% Specify the file names
# results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/robustness_test'
# if not os.path.exists(results_output_dir):
#         os.mkdir(results_output_dir)

# results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
# das_pick_file_folder = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events'
# das_pick_file_name = 'calibrated_peak_amplitude.csv'
# region_label = 'lax'
# min_channel = 100
# apply_calibrated_distance = True
# #%% load the peak amplitude results
# # Load the peak amplitude results for regression
# snr_threshold = 10
# M_threshold = [1, 10]

# peak_amplitude_df_all, DAS_index = load_and_add_region(das_pick_file_folder + '/' + das_pick_file_name, 
#                                                    region_label=region_label, snr_threshold=snr_threshold)

# peak_amplitude_df_all = peak_amplitude_df_all[(peak_amplitude_df_all.magnitude >= M_threshold[0]) & 
#                                               (peak_amplitude_df_all.magnitude <= M_threshold[1])]
                                              
# # if apply_calibrated_distance:
# #     peak_amplitude_df_predict['distance_in_km'] = peak_amplitude_df_predict['calibrated_distance_in_km']

# # Preprocessing the peak amplitude data
# #peak_amplitude_df = peak_amplitude_df.dropna()
# peak_amplitude_df_all = add_event_label(peak_amplitude_df_all)

# # filter by the channel number
# peak_amplitude_df_all = filter_by_channel_number(peak_amplitude_df_all, min_channel)
# selected_event_id = peak_amplitude_df_all.event_id.unique()

# #%% ==========================================================================================
# # Regression with weight
# given_coefficients_list = [[np.nan, np.nan, 0.590, -1.304],
#                             [np.nan, np.nan, 0.595, -1.320],
#                             [np.nan, np.nan, 0.608, -1.358],
#                             [np.nan, np.nan, 0.619, -1.393]]

# nearby_channel_number_list = [100]
# peak_amplitude_df_all = combined_channels(DAS_index, peak_amplitude_df_all, nearby_channel_number_list[0])

# # Randomly choose events, calculate the site terms
# good_proportion_all = []
# for num_fit_events in range(2, 31):
#     # i_test = 0
#     good_proportion = []
#     for i_test in range(0, 50):
#         plt.close('all')
#         event_id_fit = np.random.choice(selected_event_id, num_fit_events)
#         peak_amplitude_df_fit = peak_amplitude_df_all[peak_amplitude_df_all.event_id.isin(event_id_fit)]
#         peak_amplitude_df_predict = peak_amplitude_df_all.drop(index=peak_amplitude_df_fit.index)

#         regression_results_dir = results_output_dir + f'/robustness_test/{num_fit_events}_for_fitting'
#         if not os.path.exists(regression_results_dir):
#             os.mkdir(regression_results_dir)

#         regression_results_dir = regression_results_dir + f'/{i_test}th_test'
#         if not os.path.exists(regression_results_dir):
#             os.mkdir(regression_results_dir)

#         regression_parameter_txt = regression_results_dir + '/regression_slopes'
#         mag_slopeP, dist_slopeP, mag_slopeS, dist_slopeS = [], [], [], []

#         for ii, nearby_channel_number in enumerate(nearby_channel_number_list):
#             #peak_amplitude_df_fit = combined_channels(DAS_index, peak_amplitude_df_fit, nearby_channel_number)
#             combined_channel_id = np.sort(peak_amplitude_df_fit.combined_channel_id.unique())
            
#             given_coefficients = given_coefficients_list[ii]
#             try:
#                 regP, regS = fit_regression_with_weight_magnitude_range(peak_amplitude_df_fit, M_threshold, regression_results_dir, 
#                             nearby_channel_number, given_coefficients=given_coefficients)
                
#                 regP = None
#                 second_calibration = secondary_site_calibration(regP, regS, peak_amplitude_df_fit, given_coefficients=given_coefficients)
#                 second_calibration.to_csv(regression_results_dir + f'/secondary_site_terms_calibration_{nearby_channel_number}chan.csv', index=False)

#                 temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
#                 temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])
#             except:
#                 print('Bad data splitting, skip this time')
#                 good_proportion.append(np.nan)
#                 continue

#             try:
#                 site_term_P = regP.params[:]
#                 # temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
#                 temp_df_P['combined_channel_id'] = [int(temp.replace('C(combined_channel_id)[', '').replace(']', '')) for temp in site_term_P.index]
#                 temp_df_P['site_term_P'] = np.array(site_term_P)
#             except:
#                 print('P regression not found, assign Nan to the site term')
#                 site_term_P = np.nan
#             try:
#                 site_term_S = regS.params[:]
#                 # temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])
#                 temp_df_S['combined_channel_id'] = [int(temp.replace('C(combined_channel_id)[', '').replace(']', '')) for temp in site_term_S.index]
#                 temp_df_S['site_term_S'] = np.array(site_term_S)
#             except:
#                 print('S regression not found, assign Nan to the site term')
#                 site_term_S = np.nan


#             # combining the site terms into one csv file
#             temp_df1 = pd.DataFrame(columns=['combined_channel_id'])
#             temp_df1['combined_channel_id'] = combined_channel_id
#             temp_df1 = pd.merge(temp_df1, temp_df_P, on='combined_channel_id', how='outer')
#             temp_df1 = pd.merge(temp_df1, temp_df_S, on='combined_channel_id', how='outer')

#             temp_df2 = peak_amplitude_df_fit.loc[:, ['combined_channel_id', 'channel_id']]
#             temp_df2 = temp_df2.drop_duplicates(subset=['channel_id']).sort_values(by='channel_id')
#             site_term_df = pd.merge(temp_df2, temp_df1, on='combined_channel_id')

#             site_term_df = pd.merge(site_term_df, second_calibration, on=['channel_id'])

#             # Store the site terms
#             site_term_df.to_csv(regression_results_dir + f'/site_terms_{i_test}th_test.csv', index=False)

#             # =====================================validate strain rate
#             # only keep the channels that have been constrained
#             site_term_column='combined_channel_id'
#             site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df_all[site_term_column]])
#             ii_channel = np.isin(site_term_keys, regS.params.keys())
#             peak_amplitude_df_all_temp = peak_amplitude_df_all[ii_channel]

#             peak_amplitude_df_all_temp = pd.merge(peak_amplitude_df_all_temp, second_calibration, on=['channel_id', 'region'])
#             peak_amplitude_calculated = regS.predict(peak_amplitude_df_all_temp)

#             given_coefficients = given_coefficients_list[ii]
#             mag_coef_S, dist_coef_S = given_coefficients[2], given_coefficients[3]
#             peak_amplitude_calculated = peak_amplitude_calculated + (peak_amplitude_df_all_temp.magnitude * mag_coef_S) \
#                                         + np.log10(peak_amplitude_df_all_temp.distance_in_km)*dist_coef_S

#             peak_amplitude_calculated = peak_amplitude_calculated + peak_amplitude_df_all_temp.diff_peak_S
#             peak_amplitude_calculated = 10**peak_amplitude_calculated


#             temp_peaks = np.array([np.array(peak_amplitude_df_all_temp.peak_S), peak_amplitude_calculated]).T
#             peak_comparison_df = pd.DataFrame(data=temp_peaks,
#                                         columns=['peak_S', 'peak_S_predict'])

#             g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 100], phase='S')
#             g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{nearby_channel_number}chan_seaborn.png')
#             g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{nearby_channel_number}chan_seaborn.pdf')


#             # =====================================predict
#             # only keep the channels that have been constrained
#             site_term_column='combined_channel_id'
#             site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df_fit[site_term_column]])
#             ii_channel = np.isin(site_term_keys, regS.params.keys())
#             peak_amplitude_df_fit = peak_amplitude_df_fit[ii_channel]
#             # Apply to the fit set
#             M_fit, peak_amplitude_df_fit_temp = calculate_magnitude_from_strain(peak_amplitude_df_fit, regS, type='S', 
#                 site_term_column=site_term_column, given_coefficients=given_coefficients_list[ii][-2:], secondary_calibration=True)
#             M_df_fit = get_mean_magnitude(peak_amplitude_df_fit_temp, M_fit)

#             # Apply to the prediction set
#             peak_amplitude_df_predict = combined_channels(DAS_index, peak_amplitude_df_predict, nearby_channel_number)
#             site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df_predict[site_term_column]])
#             ii_channel = np.isin(site_term_keys, regS.params.keys())
#             peak_amplitude_df_predict = peak_amplitude_df_predict[ii_channel]
    
#             M_predict, peak_amplitude_df_predict_temp = calculate_magnitude_from_strain(peak_amplitude_df_predict, regS, type='S', 
#                 site_term_column=site_term_column, given_coefficients=given_coefficients_list[ii][-2:], secondary_calibration=True)
#             M_df_predict = get_mean_magnitude(peak_amplitude_df_predict_temp, M_predict)

#             gca = plot_magnitude_seaborn(M_df_predict)
#             gca.ax_joint.plot(M_df_fit.magnitude, M_df_fit.predicted_M, 'k.', markersize=10)
#             gca.savefig(regression_results_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.png",bbox_inches='tight')
#             gca.savefig(regression_results_dir + f"/predicted_magnitude_S_{nearby_channel_number}_seaborn.pdf",bbox_inches='tight')
#             plt.close('all')

#             # count the percentage of good estimation in the predicted dataset
#             M_error = M_df_predict.magnitude - M_df_predict.predicted_M
#             good_proportion.append(len(M_error[abs(M_error) <=0.5])/len(M_error))

#     good_proportion_all.append(good_proportion)

# good_proportion_all = np.array(good_proportion_all)
# num_fit_events = np.arange(2, 31)
# np.savez(results_output_dir + f'/robustness_test/good_proportion_{i_test}.npz', num_fit_events=num_fit_events, good_proportion_all=good_proportion_all)


# #%% Compare site term
# colors = plt.cm.viridis(np.linspace(0,1,50))
# channel_space = 10/1e3
# num_fit_events = 10
# site_term_std_list = []
# for num_fit_events in range(2, 31):

#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(10, 6), sharex=True, gridspec_kw={})

#     gca = ax
    
#     for i_model, combined_channel_number in enumerate(nearby_channel_number_list):
#         channel_all = np.arange(0, 10000)
#         site_term_channel_all = np.zeros((50, 10000))*np.nan
#         for i_test in range(50):
#             try:
#                 site_term_df = pd.read_csv(results_output_dir + f'/robustness_test/{num_fit_events}_for_fitting/{i_test}th_test/site_terms_{i_test}th_test.csv')
                
#                 # gca.plot(site_term_df.channel_id, site_term_df.site_term_P, '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
#                 gca.plot(site_term_df.channel_id*channel_space, site_term_df.site_term_S + site_term_df.diff_peak_S, '.', color=colors[i_test], zorder=-i_test, alpha=1, linewidth=1)#, Cond.# {regS.condition_number:.2f}')
#                 #gca.plot(site_term_df.channel_id*channel_space, site_term_df.site_term_S, '--', color=colors[i_test], zorder=-i_test, alpha=0.9, linewidth=1)#, Cond.# {regS.condition_number:.2f}')
                
#                 site_term_channel_all[i_test, site_term_df.channel_id.astype('int')] = np.array(site_term_df.site_term_S + site_term_df.diff_peak_S)
#             except:
#                 continue
#     site_term_mean = np.nanmean(site_term_channel_all, axis=0)
#     site_term_std_list.append(np.nanmean(np.nanstd(site_term_channel_all, axis=0)))

#     gca.plot(channel_all*channel_space, site_term_mean, '-r', alpha=1, linewidth=1)

#     # gca.set_yscale('log')
#     gca.set_ylabel('Site terms S (in log10)')
#     gca.set_xlabel('Distance (km)')
#     gca.grid()

#     #gca.set_xticklabels(np.arange(0, 10000, 1000))
#     gca.set_ylim(-2.5, 1)
#     gca.set_title(f'{num_fit_events} events for site term calibration')
#     # gca.legend(loc=3, fontsize=14)

#     plt.savefig(results_output_dir + f'/robustness_test/site_terms_variation_with_event_number/site_terms_{num_fit_events}_events_to_fit.png', bbox_inches='tight')

# #%% show site term std variation
# fig, gca = plt.subplots(figsize=(10, 6))
# gca.plot(range(2, 31), site_term_std_list, '-ro', linewidth=2, label='STD')
# gca.grid()
# gca.legend(loc=1)
# gca.set_xlim(0, 32)
# gca.set_xlabel('Number of events for calibration')
# gca.set_ylabel('Site term STD')
# plt.savefig(results_output_dir + f'/robustness_test/site_terms_variation_with_event_number/site_term_STD_variation.png', bbox_inches='tight')
# plt.savefig(results_output_dir + f'/robustness_test/site_terms_variation_with_event_number/site_term_STD_variation.pdf', bbox_inches='tight')

# #%% show how well the magnitude is estimated
# tempx = np.load(results_output_dir + f'/robustness_test/good_proportion_49.npz', allow_pickle=True)
# good_proportion_all = tempx['good_proportion_all']
# num_fit_events = tempx['num_fit_events']

# mean_percentage = np.zeros(len(good_proportion_all))

# fig, gca = plt.subplots(figsize=(10, 6))
# for ii in range(len(good_proportion_all)):
#     percentage_temp = 100*np.array(good_proportion_all[ii])
#     percentage_temp = np.reshape(percentage_temp, [1, -1])

#     gca.plot(num_fit_events[ii], percentage_temp, 'k.', alpha=0.6)
#     mean_percentage[ii] = np.mean(percentage_temp)

# gca.plot(np.nan, np.nan, 'k.', alpha=0.6, label='each random test')
# gca.plot(range(2, 31), mean_percentage, '-r', linewidth=5, label='Mean')
# gca.grid()
# gca.legend(loc=4)
# gca.set_xlim(0, 32)
# gca.set_xlabel('Number of events for calibration')
# gca.set_ylabel('Percentage (%)')
# gca.set_title('Percentage of good magnitude estimation ($\pm 0.5$)')
# plt.savefig(results_output_dir + f'/magnitude_estimation_performance.png', bbox_inches='tight')
# plt.savefig(results_output_dir + f'/magnitude_estimation_performance.pdf', bbox_inches='tight')



# # %%
