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
# ==============================  Looking into the multiple array case ========================================
# Specify the file names
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'

min_channel = 100
snr_threshold = 10
magnitude_threshold = [2, 10]
apply_calibrated_distance = True # if true, use the depth-calibrated distance to do regression
output_label = ''
apply_secondary_calibration = False # if true, use the secondary site term calibration

#%%
combined_channel_number_list = [10, 100]
regression_results_dir = results_output_dir + f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

regression_parameter_txt = regression_results_dir + '/regression_slopes'
reg_P_list = []

for nearby_channel_number in combined_channel_number_list:
    # Load the processed DataFrame
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
    reg_P_list.append(regP)

#%% 
# compare site terms in details
site_term_df1 = pd.read_csv(regression_results_dir + f'/site_terms_{10}chan.csv')
site_term_df2 = pd.read_csv(regression_results_dir + f'/site_terms_{100}chan.csv')

y_P_predict1 = reg_P_list[0].predict(peak_amplitude_df)
y_P_predict2 = reg_P_list[1].predict(peak_amplitude_df)

#%%
peak_amplitude_df1 = peak_amplitude_df.copy()
peak_amplitude_df1['predict_y_P'] = 10**y_P_predict1

peak_amplitude_df2 = peak_amplitude_df.copy()
peak_amplitude_df2['predict_y_P'] = 10**y_P_predict2


fig, ax = plt.subplots(figsize=(10, 10))
ii = (peak_amplitude_df.region == 'mammothN') & (peak_amplitude_df.channel_id == 2425) #2423 2424 2425
ax.plot(np.log10(peak_amplitude_df2[ii].peak_P), np.log10(peak_amplitude_df2[ii].predict_y_P), 'r.', alpha=1, label='100')
ax.plot(np.log10(peak_amplitude_df1[ii].peak_P), np.log10(peak_amplitude_df1[ii].predict_y_P), 'b.', alpha=1, label='10')
ax.plot([-5, 3], [-5, 3], '-k')
ax.plot([-5, 3], [-4, 4], '--k')
ax.plot([-5, 3], [-6, 2], '--k')
ax.legend()
ax.set_xlim(-3, 2)
ax.set_ylim(-3, 2)

peak_amplitude_df2[ii]


#%%
fig, ax = plt.subplots(2, 1, figsize=(16, 8), squeeze=False, sharex=True)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
title_plot = ['Ridgecrest P', 'Ridgecrest S', 'Long Valley North P', 'Long Valley North S', 'Long Valley South P', 'Long Valley South S']
region_labels = ['ridgecrest', 'mammothN', 'mammothS']
channel_spacing = [8e-3, 10e-3, 10e-3]
label_lists = ['10 channels','20 channels','50 channels','100 channels','constant']
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model


region_labels = ['mammothN']
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')

    for i_region, region_label in enumerate(region_labels):
        site_term_df_now = site_term_df[site_term_df.region_site.str.contains(region_label)]
        site_term_df_now = site_term_df_now.sort_values(by='channel_id')
        
        gca = ax[0, i_region]
        if i_region != 2: 
            gca.plot(site_term_df_now.channel_id, site_term_df_now.site_term_P)
        else:
            gca.plot(site_term_df_now.channel_id, site_term_df_now.site_term_P, label=label_lists[i_model])
        gca.set_title(title_plot[i_region * 2])
        #gca.grid()

        gca = ax[1, i_region]
        gca.plot(site_term_df_now.channel_id, site_term_df_now.site_term_S)
        gca.set_title(title_plot[i_region * 2 + 1])
        #gca.grid()
        #gca.sharey(ax[i_region, 0])

gca.set_xlim(2400, 2500)

#%%
site_term_df1[site_term_df1.channel_id==2425], site_term_df2[site_term_df2.channel_id==2425]

#%%
peak_amplitude_df_temp = pd.DataFrame(columns=['region_site', 'channel_id', 'region', 'magnitude', 'site_std', 'diff_peak_P', 'diff_peak_S', 'site_calibate_P', 'site_calibrate_S'])
peak_amplitude_df_temp.channel_id = peak_amplitude_df.channel_id
peak_amplitude_df_temp.region = peak_amplitude_df.region
peak_amplitude_df_temp.magnitude = peak_amplitude_df.magnitude

#peak_amplitude_df_temp.site_std = -y_P_predict1 + np.log10(peak_amplitude_df.peak_P)

weighted_all = np.nansum(10**peak_amplitude_df.magnitude)
peak_amplitude_df_temp.diff_peak_P = (-y_P_predict1 + np.log10(peak_amplitude_df.peak_P))#*10**peak_amplitude_df.magnitude/weighted_all
# peak_amplitude_df_temp.diff_peak_S = (-y_S_predict1 + np.log10(peak_amplitude_df.peak_S))*10**peak_amplitude_df.magnitude/weighted_all
peak_amplitude_df_temp.region_site = peak_amplitude_df_temp.region + '-' + peak_amplitude_df.combined_channel_id.astype('str')

# ridgecrest_second_calibration = peak_amplitude_df_temp[peak_amplitude_df_temp.region == 'ridgecrest'].groupby(['channel_id'], as_index=False).mean()
# mammothN_second_calibration = peak_amplitude_df_temp[peak_amplitude_df_temp.region == 'mammothN'].groupby(['channel_id'], as_index=False).mean()
# mammothS_second_calibration = peak_amplitude_df_temp[peak_amplitude_df_temp.region == 'mammothS'].groupby(['channel_id'], as_index=False).mean()

second_calibration = peak_amplitude_df_temp.groupby(['channel_id', 'region'], as_index=False).mean()
temp_df = peak_amplitude_df_temp[['region_site', 'channel_id', 'region']].drop_duplicates(subset=['channel_id', 'region_site'])
second_calibration = pd.merge(second_calibration, temp_df, on=['channel_id', 'region'])
second_calibration = second_calibration.drop(columns=['magnitude'])

second_calibration = second_calibration.sort_values(by=['channel_id'])
#%%
site_term_df1 = site_term_df1.sort_values(by='channel_id')
ii_region1 = site_term_df1.region_site.str.contains('mammothS')
plt.plot(site_term_df1[ii_region1].channel_id, site_term_df1[ii_region1].site_term_P, '-')

ii_region = second_calibration.region == 'mammothS'
#plt.plot(site_term_df1[ii_region1].channel_id, np.array(site_term_df1[ii_region1].site_term_P), '-', zorder=10)
plt.plot(site_term_df1[ii_region1].channel_id, np.array(site_term_df1[ii_region1].site_term_P)+np.array(second_calibration[ii_region].diff_peak_P), '.')
#plt.xlim(2200, 2400)

#%%

peak_amplitude_df_xx = pd.merge(peak_amplitude_df, second_calibration, on=['channel_id', 'region', 'region_site'])

y_P_predict2 = reg_P_list[0].predict(peak_amplitude_df_xx)
y_P_predict2_calibrated = reg_P_list[0].predict(peak_amplitude_df_xx)+peak_amplitude_df_xx.diff_peak_P

#%%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(np.log10(peak_amplitude_df_xx.peak_P), y_P_predict2, 'r.', alpha=0.01)
ax.plot(np.log10(peak_amplitude_df_xx.peak_P), y_P_predict2_calibrated, 'b.', alpha=0.01)
ax.plot([-5, 3], [-5, 3], '-k')
ax.plot([-5, 3], [-4, 4], '--k')
ax.plot([-5, 3], [-6, 2], '--k')
ax.legend()
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)
# %%
ii_one_channel = peak_amplitude_df_xx.channel_id == 2425
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(np.log10(peak_amplitude_df_xx[ii_one_channel].peak_P), y_P_predict2[ii_one_channel], 'r.', alpha=1)
ax.plot(np.log10(peak_amplitude_df_xx[ii_one_channel].peak_P), y_P_predict2_calibrated[ii_one_channel], 'b.', alpha=1)
ax.plot([-5, 3], [-5, 3], '-k')
ax.plot([-5, 3], [-4, 4], '--k')
ax.plot([-5, 3], [-6, 2], '--k')
ax.legend()
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)
# %%
