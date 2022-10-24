#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser
import obspy
import statsmodels.api as sm

from utility.general import mkdir
from utility.processing import remove_outliers, filter_event

import seaborn as sns

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %matplotlib inline
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 18, # fontsize for x and y labels (was 10)
    'axes.titlesize': 18,
    'font.size': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex':False,
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
}
matplotlib.rcParams.update(params)

#%% 
#Calculate the possible clipping lines 
combined_results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'

# maximum clipping amplitude
peak_amplitude_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South',
                           '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate']

region_keys = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']

max_range = [110, 30, 30, 30]

a_M_P, b_D_P = {}, {}
a_M_S, b_D_S = {}, {}
region_clip, region_sense = {}, {}
site_term_P_mean, site_term_S_mean = {}, {}
M_clipping = np.arange(0, 9)

# dictionary to store clipping relation
D_clipping_P, D_clipping_S = {}, {}
# dictionary to store sensible distance relation
D_sense_P, D_sense_S = {}, {}

for ii_region in [0, 1, 2, 3]:

    peak_amplitude_dir = peak_amplitude_dir_list[ii_region]

    # load regression results 
    regP_pre_path = peak_amplitude_dir + f"/iter_regression_results_smf_weighted_100_channel_at_least/P_regression_combined_site_terms_iter.pickle"
    regS_pre_path = peak_amplitude_dir + f"/iter_regression_results_smf_weighted_100_channel_at_least/S_regression_combined_site_terms_iter.pickle"

    # load coefficents
    try:
        regP_pre = sm.load(regP_pre_path)
        a_M_P[region_keys[ii_region]] = regP_pre.params['magnitude']
        b_D_P[region_keys[ii_region]] = regP_pre.params['np.log10(distance_in_km)']
    except:
        a_M_P[region_keys[ii_region]] = np.nan
        b_D_P[region_keys[ii_region]] = np.nan

    try:
        regS_pre = sm.load(regS_pre_path)
        a_M_S[region_keys[ii_region]] = regS_pre.params['magnitude']
        b_D_S[region_keys[ii_region]] = regS_pre.params['np.log10(distance_in_km)']
    except:
        a_M_S[region_keys[ii_region]] = np.nan
        b_D_S[region_keys[ii_region]] = np.nan       

    # load measured clip amplitude
    temp = np.load(peak_amplitude_dir + '/peak_amplitude_events/global_maximum.npz')
    temp = temp['array_maximum']
    if region_keys[ii_region] == 'mammothS':
        region_sense[region_keys[ii_region]] = np.log10(np.amin(temp[temp>10**-1.8]))
    elif region_keys[ii_region] == 'mammothN':
        region_sense[region_keys[ii_region]] = np.log10(np.amin(temp[temp>10**-2]))
    else:
        region_sense[region_keys[ii_region]] = np.log10(np.amin(temp[temp>0]))
    region_clip[region_keys[ii_region]] = np.log10(np.amax(temp[temp<=max_range[ii_region]]))

    # load site terms
    # load and calculate average site term for each region
    site_term_file = peak_amplitude_dir + f"/iter_regression_results_smf_weighted_100_channel_at_least/site_terms_iter.csv"
    site_term_mean = pd.read_csv(site_term_file)
    site_term_P_mean[region_keys[ii_region]] = site_term_mean.site_term_P.mean()
    site_term_S_mean[region_keys[ii_region]] = site_term_mean.site_term_S.mean()

    # P and S clipping distance relation
    D_clipping_P[region_keys[ii_region]] = (-M_clipping*a_M_P[region_keys[ii_region]] + (region_clip[region_keys[ii_region]]-site_term_P_mean[region_keys[ii_region]]))/b_D_P[region_keys[ii_region]]
    D_clipping_S[region_keys[ii_region]] = (-M_clipping*a_M_S[region_keys[ii_region]] + (region_clip[region_keys[ii_region]]-site_term_S_mean[region_keys[ii_region]]))/b_D_S[region_keys[ii_region]]

    # P and S sensible distance relation
    D_sense_P[region_keys[ii_region]] = (-M_clipping*a_M_P[region_keys[ii_region]] + (region_sense[region_keys[ii_region]]-site_term_P_mean[region_keys[ii_region]]))/b_D_P[region_keys[ii_region]]
    D_sense_S[region_keys[ii_region]] = (-M_clipping*a_M_S[region_keys[ii_region]] + (region_sense[region_keys[ii_region]]-site_term_S_mean[region_keys[ii_region]]))/b_D_S[region_keys[ii_region]]


#%% 
# Combine DataFrame into one DataFrame
for ii_region in [0, 1, 2, 3]:
    peak_amplitude_dir = peak_amplitude_dir_list[ii_region]
    # load and concatinate peak_amplitude_df
    if ii_region == 0:
        peak_amplitude_df_all = pd.read_csv(peak_amplitude_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
    else:
        peak_amplitude_df_temp = pd.read_csv(peak_amplitude_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
        peak_amplitude_df_all = pd.concat((peak_amplitude_df_all, peak_amplitude_df_temp), axis=0, ignore_index=True)

#%%
# Show the detectable figure BEFORE event filter
# plot the M-D distribution for each region
peak_amplitude_df_temp = peak_amplitude_df_all.iloc[::1, :]
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
# peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P.astype('float'))
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float'))
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S
# group by events
peak_amplitude_df_temp = peak_amplitude_df_temp.groupby(by=['event_id', 'region'], as_index=False).mean()

fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
fig.suptitle("Before", fontsize=20)
ax = ax.flatten()

region_keys = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']
region_title = ['Ridgecrest', 'Long Valley N', 'Long Valley S', 'Sanriku']
color_list = ['#7DC0A6', '#ED926B', '#91A0C7', '#DA8EC0']


for ii_region, region_key in enumerate(region_keys):
    peak_amplitude_df_current = peak_amplitude_df_temp[peak_amplitude_df_temp.region == region_key]

    gca = ax[ii_region]
    cbaxes = inset_axes(gca, width="3%", height="30%", loc=2) 

    g = sns.histplot(ax=gca, data=peak_amplitude_df_current[['magnitude','log10(distance)']], x='magnitude', y='log10(distance)',
                 color=".1", pmax=0.7, cbar=True, bins=(20, 20), cbar_ax=cbaxes, cbar_kws={'fraction':0.03 ,'pad':0.0, 'label':'# of EQs'})

    gca.fill_between(M_clipping,D_clipping_P[region_key], D_sense_P[region_key], label='P detectable range',
                     color=color_list[ii_region], linestyle='--', edgecolor='k', alpha=0.2)
    gca.fill_between(M_clipping,D_clipping_S[region_key], D_sense_S[region_key], label='S detectable range',
                     color=color_list[ii_region], linestyle='-', edgecolor='k', alpha=0.2)
    gca.set_title(region_title[ii_region])

    if region_key == 'ridgecrest':
        temp_clip = peak_amplitude_df_current[peak_amplitude_df_current.event_id.astype('int').isin([38548295, 39462536])]
        gca.plot(temp_clip.magnitude-0.05, temp_clip['log10(distance)'], 'rx')

    if (region_key == 'mammothN') or (region_key == 'mammothS'):
        temp_clip = peak_amplitude_df_current[peak_amplitude_df_current.event_id.astype('int').isin([73584926])]
        gca.plot(temp_clip.magnitude-0.05, temp_clip['log10(distance)'], 'rx')

    if region_key == 'sanriku':
        temp_clip = peak_amplitude_df_current[peak_amplitude_df_current.event_id.astype('int').isin([4130])]
        gca.plot(temp_clip.magnitude-0.05, temp_clip['log10(distance)'], 'rx')

    gca.legend(fontsize=12, loc=4)

# some final handling
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for i_ax, gca in enumerate(ax.flatten()):
    gca.spines.right.set_visible(False)
    gca.spines.top.set_visible(False)
    # add annotation
    gca.annotate(f'({letter_list[k]})', xy=(-0.1, 1.1), xycoords=gca.transAxes)
    k += 1

plt.savefig(combined_results_output_dir + '/data_correlation_detectable_range_one_all_data.png', bbox_inches='tight')
#plt.savefig('/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/data_correlation_clipping_one.png', bbox_inches='tight')


# %%
# Show the detectable figure AFTER event filter
# plot the M-D distribution for each region

# apply event filter (directly load the filtered events)
sanriku_results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'

snr_threshold = 10
min_channel = 100
magnitude_threshold = [2, 10]
snr_threshold_Sanriku = 5

peak_amplitude_df_temp = pd.read_csv(combined_results_output_dir + '/peak_amplitude_multiple_arrays.csv')
peak_amplitude_df_sanriku = pd.read_csv(sanriku_results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
peak_amplitude_df_sanriku = filter_event(peak_amplitude_df_sanriku, snr_threshold=snr_threshold_Sanriku, M_threshold=magnitude_threshold, min_channel=min_channel)
peak_amplitude_df_temp = pd.concat((peak_amplitude_df_temp, peak_amplitude_df_sanriku), axis=0, ignore_index=True)


# plot the M-D distribution for each region
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
# peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P.astype('float'))
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float'))
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S
peak_amplitude_df_temp = peak_amplitude_df_temp.groupby(by=['event_id', 'region'], as_index=False).mean()

fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
fig.suptitle("After", fontsize=20)
ax = ax.flatten()

region_keys = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']
region_title = ['Ridgecrest', 'Long Valley N', 'Long Valley S', 'Sanriku']
color_list = ['#7DC0A6', '#ED926B', '#91A0C7', '#DA8EC0']

for ii_region, region_key in enumerate(region_keys):
    peak_amplitude_df_current = peak_amplitude_df_temp[peak_amplitude_df_temp.region == region_key]
    
    if region_key == 'sanriku': # not show 4130
        peak_amplitude_df_current = peak_amplitude_df_current.drop(index=peak_amplitude_df_current[peak_amplitude_df_current.event_id == 4130].index)

    gca = ax[ii_region]
    cbaxes = inset_axes(gca, width="3%", height="30%", loc=2) 

    g = sns.histplot(ax=gca, data=peak_amplitude_df_current[['magnitude','log10(distance)']], x='magnitude', y='log10(distance)',
                 color=".1", pmax=0.7, cbar=True, bins=(20, 10), cbar_ax=cbaxes, cbar_kws={'fraction':0.03 ,'pad':0.0, 'label':'# of EQs'})

    gca.fill_between(M_clipping,D_clipping_P[region_key], D_sense_P[region_key], label='P detectable range',
                     color=color_list[ii_region], linestyle='--', edgecolor='k', alpha=0.2)
    gca.fill_between(M_clipping,D_clipping_S[region_key], D_sense_S[region_key], label='S detectable range',
                     color=color_list[ii_region], linestyle='-', edgecolor='k', alpha=0.2)
    gca.set_title(region_title[ii_region])

    if region_key == 'ridgecrest':
        temp_clip = peak_amplitude_df_current[peak_amplitude_df_current.event_id.isin([38548295.0, 39462536.0])]
        gca.plot(temp_clip.magnitude-0.05, temp_clip['log10(distance)'], 'rx')
    if (region_key == 'mammothN') or (region_key == 'mammothS'):
        temp_clip = peak_amplitude_df_current[peak_amplitude_df_current.event_id.astype('int').isin([73584926])]
        gca.plot(temp_clip.magnitude-0.05, temp_clip['log10(distance)'], 'rx')
    if region_key == 'sanriku':
        temp_4130 = peak_amplitude_df_current[peak_amplitude_df_current.event_id == 4130]
        gca.plot(temp_4130.magnitude-0.05, temp_4130['log10(distance)'], 'r.')

    gca.legend(fontsize=12, loc=4)

# some final handling
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for i_ax, gca in enumerate(ax.flatten()):
    gca.spines.right.set_visible(False)
    gca.spines.top.set_visible(False)
    # add annotation
    gca.annotate(f'({letter_list[k]})', xy=(-0.1, 1.1), xycoords=gca.transAxes)
    k += 1

plt.savefig(combined_results_output_dir + '/data_correlation_detectable_range_one_filtered_data.png', bbox_inches='tight')
#plt.savefig('/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/data_correlation_clipping_one.png', bbox_inches='tight')


# %%
