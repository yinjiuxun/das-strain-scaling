#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser
import obspy
import statsmodels.api as sm

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import remove_outliers, filter_event
from utility.plotting import add_annotate

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
def plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax, show_region=True, show_clip=True):
    region_keys = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']
    region_title = ['Ridgecrest', 'Long Valley N', 'Long Valley S', 'Sanriku']
    color_list = ['#7DC0A6', '#ED926B', '#91A0C7', '#DA8EC0']


    for ii_region, region_key in enumerate(region_keys):
        peak_amplitude_df_current = peak_amplitude_df_temp[peak_amplitude_df_temp.region == region_key]

        gca = ax[ii_region]
        cbaxes = inset_axes(gca, width="3%", height="30%", loc=2) 

        g = sns.histplot(ax=gca, data=peak_amplitude_df_current[['magnitude','log10(distance)']], x='magnitude', y='log10(distance)',
                 color=".1", pmax=0.7, cbar=True, bins=(20, 20), cbar_ax=cbaxes, cbar_kws={'fraction':0.03 ,'pad':0.0, 'label':'# of EQs'})

        if show_region:
            alpha = 0.2
        else:
            alpha = 0.0

        gca.fill_between(M_clipping,D_clipping_P[region_key], D_sense_P[region_key], label='P detectable range',
                    color=color_list[ii_region], linestyle='--', edgecolor='k', alpha=alpha)
        gca.fill_between(M_clipping,D_clipping_S[region_key], D_sense_S[region_key], label='S detectable range',
                    color=color_list[ii_region], linestyle='-', edgecolor='k', alpha=alpha)

        if show_region:
            gca.legend(fontsize=12, loc=4)

        gca.set_title(region_title[ii_region])

        if show_clip:
            if region_key == 'ridgecrest':
                temp_clip = peak_amplitude_df_current[peak_amplitude_df_current.event_id.astype('int').isin([38548295, 39462536])]
                gca.plot(temp_clip.magnitude-0.05, temp_clip['log10(distance)'], 'rx')

            if (region_key == 'mammothN') or (region_key == 'mammothS'):
                temp_clip = peak_amplitude_df_current[peak_amplitude_df_current.event_id.astype('int').isin([73584926])]
                gca.plot(temp_clip.magnitude-0.05, temp_clip['log10(distance)'], 'rx')

            if region_key == 'sanriku':
                temp_clip = peak_amplitude_df_current[peak_amplitude_df_current.event_id.astype('int').isin([4130])]
                gca.plot(temp_clip.magnitude-0.05, temp_clip['log10(distance)'], 'rx')

        
# some final handling
    letter_list = [str(chr(k+97)) for k in range(0, 20)]
    k=0
    for i_ax, gca in enumerate(ax.flatten()):
        gca.spines.right.set_visible(False)
        gca.spines.top.set_visible(False)
    # add annotation
        gca.annotate(f'({letter_list[k]})', xy=(-0.1, 1.1), xycoords=gca.transAxes)
        k += 1

    return fig, ax
#%% Calculate the possible clipping lines 
#Calculate the possible clipping lines 
combined_results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
figure_output_dir = combined_results_output_dir + '/data_figures'
mkdir(figure_output_dir)
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


#%% Concatenate dataset if needed
reconcatenate_data = False
if reconcatenate_data:
# Combine DataFrame into one DataFrame
    for ii_region in [0, 1, 2]:
        peak_amplitude_dir = peak_amplitude_dir_list[ii_region]
        # load and concatinate peak_amplitude_df
        if ii_region == 0:
            peak_amplitude_df_all = pd.read_csv(peak_amplitude_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
        else:
            peak_amplitude_df_temp = pd.read_csv(peak_amplitude_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
            peak_amplitude_df_all = pd.concat((peak_amplitude_df_all, peak_amplitude_df_temp), axis=0, ignore_index=True)

    peak_amplitude_df_all.to_csv(figure_output_dir + '/peak_amplitude_multiple_arrays_all_raw.csv')

#%%
# this is only for presentation purpose
show_region = False # this is only for presentation purpose

#%% show all events
# show all events
sanriku_results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'

peak_amplitude_df_all = pd.read_csv(combined_results_output_dir + '/peak_amplitude_multiple_arrays_all_raw.csv')
peak_amplitude_df_sanriku = pd.read_csv(sanriku_results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')

peak_amplitude_df_all = pd.concat((peak_amplitude_df_all, peak_amplitude_df_sanriku), axis=0, ignore_index=True)

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
fig.suptitle("All events in the dataset", fontsize=20)
ax = ax.flatten()
fig, ax = plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax)
plt.savefig(figure_output_dir + '/data_correlation_detectable_range_one_all_data.png', bbox_inches='tight')
#plt.savefig('/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/data_correlation_clipping_one.png', bbox_inches='tight')

if not show_region:
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle(f"All events in the dataset", fontsize=20)
    ax = ax.flatten()
    fig, ax = plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax, 
                                    show_region=show_region, show_clip=False)
    plt.savefig(figure_output_dir + '/data_correlation_detectable_range_one_all_data_present.png', bbox_inches='tight')


#%% show data after min_channel filter
# show data after min_channel filter
peak_amplitude_df_all = filter_event(peak_amplitude_df_all, min_channel=100)

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
fig.suptitle(f"Events with more than 100 available channels", fontsize=20)
ax = ax.flatten()
fig, ax = plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax)
plt.savefig(figure_output_dir + '/data_correlation_detectable_range_one_all_data_100_min_channel.png', bbox_inches='tight')

if not show_region:
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle(f"Events with more than 100 available channels", fontsize=20)
    ax = ax.flatten()
    fig, ax = plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax, 
                                    show_region=False, show_clip=False)
    plt.savefig(figure_output_dir + '/data_correlation_detectable_range_one_all_data_100_min_channel_present.png', bbox_inches='tight')

#%% check the distribution of SNR and Magnitude
# check the distribution of SNR and Magnitude
# SNR
peak_amplitude_df_all = filter_event(peak_amplitude_df_all, min_channel=100)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()
for ii_region, region_key in enumerate(region_keys):
    gca = ax[ii_region]
    gca.hist(peak_amplitude_df_all[peak_amplitude_df_all.region==region_key].snrP, range=(0, 30), bins=50, label='P')
    gca.hist(peak_amplitude_df_all[peak_amplitude_df_all.region==region_key].snrS, range=(0, 30), bins=50, alpha=0.5, label='S')
    if ii_region > 1:
        gca.set_xlabel('SNR')
    if ii_region in [0, 2]:
        gca.set_ylabel('Counts')
    gca.set_title(region_key)
gca.legend()
plt.savefig(figure_output_dir + '/SNR_distribution.png', bbox_inches='tight')

# magnitude
peak_amplitude_df_temp = peak_amplitude_df_all.groupby(by=['event_id', 'region'], as_index=False).mean()
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()
for ii_region, region_key in enumerate(region_keys):
    gca = ax[ii_region]
    gca.hist(peak_amplitude_df_temp[peak_amplitude_df_temp.region==region_key].magnitude, range=(0, 8), bins=20)
    if ii_region > 1:
        gca.set_xlabel('Magnitude')
    if ii_region in [0, 2]:
        gca.set_ylabel('Counts')
    gca.set_title(region_key)
plt.savefig(figure_output_dir + '/magnitude_distribution.png', bbox_inches='tight')

#%% Show events with SNR thresholding
# Show events with SNR thresholding
peak_amplitude_df_all = filter_event(peak_amplitude_df_all, snr_threshold=5, min_channel=100)

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
fig.suptitle("Events with more than 100 available channels, SNR filtered", fontsize=20)
ax = ax.flatten()
fig, ax = plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax)
plt.savefig(figure_output_dir + '/data_correlation_detectable_range_one_SNR5_filtered_data.png', bbox_inches='tight')

if not show_region:
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle("Events with more than 100 available channels, SNR filtered", fontsize=20)
    ax = ax.flatten()
    fig, ax = plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax, 
                                    show_region=False, show_clip=False)
    plt.savefig(figure_output_dir + '/data_correlation_detectable_range_one_SNR5_filtered_data_present.png', bbox_inches='tight')


#%% show magnitude varying with mean peak amplitude for all events
# show magnitude varying with mean peak amplitude for all events
region_keys = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']
region_title = ['Ridgecrest', 'Long Valley N', 'Long Valley S', 'Sanriku']
color_list = ['#7DC0A6', '#ED926B', '#91A0C7', '#DA8EC0']

def running_median(magnitude, peak, bins=20, M_range=(0, 8)):
    magnitude = np.array(magnitude)
    peak = np.array(peak)

    magnitude_bins = np.linspace(M_range[0], M_range[1], bins)
    bin_size = magnitude_bins[1] - magnitude_bins[0]
    bin_center = magnitude_bins[:-1] + bin_size/2

    peak_median = np.zeros(len(bin_center))

    for ii in range(bins-1):
        peak_median[ii] = np.nanmedian(peak[(magnitude>=magnitude_bins[ii]) & (magnitude<magnitude_bins[ii+1])])
    return bin_center[~np.isnan(peak_median)], peak_median[~np.isnan(peak_median)]

fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
ax = ax.flatten()
for ii_region, region_key in enumerate(region_keys):
    temp_now = peak_amplitude_df_temp[peak_amplitude_df_temp.region==region_key]

    temp_now = temp_now.drop(index=temp_now[(temp_now.peak_P>=1e3)|(temp_now.peak_S>=1e3)].index)
    alpha=1

    bin_center, peak_P_median = running_median(temp_now.magnitude, temp_now.peak_P, bins=22)
    ax[0].semilogy(bin_center, peak_P_median, '-o', color=color_list[ii_region], label=region_title[ii_region], markersize=8, alpha=alpha)
    ax[0].vlines(x=2, ymin=1e-2, ymax=10, linestyle='--', color='k')

    bin_center, peak_S_median = running_median(temp_now.magnitude, temp_now.peak_S, bins=22)
    ax[1].semilogy(bin_center, peak_S_median, '-o', color=color_list[ii_region], label=region_title[ii_region], markersize=8, alpha=alpha)
    ax[1].vlines(x=2, ymin=1e-2, ymax=50, linestyle='--', color='k')


ax[0].set_title('P wave')
ax[1].legend(fontsize=10, loc=4)
ax[0].set_xlabel('Magnitude')
ax[0].set_ylabel('Median peak amplitude')
ax[1].set_title('S wave')
ax[1].set_xlabel('Magnitude')

ax = add_annotate(ax)
plt.savefig(figure_output_dir + '/magnitude_vs_mean_peak_amplitude.png', bbox_inches='tight')

#%% show magnitude varying with mean peak amplitude for all events, SCALED BY DISTANCE
# show magnitude varying with mean peak amplitude for all events
region_keys = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']
region_title = ['Ridgecrest', 'Long Valley N', 'Long Valley S', 'Sanriku']
color_list = ['#7DC0A6', '#ED926B', '#91A0C7', '#DA8EC0']

def running_median_scaled_by_distance(magnitude, peak, distance, b_D, bins=20, M_range=(0, 8)):
    magnitude = np.array(magnitude)
    peak = np.array(peak)
    distance = np.array(distance)

    magnitude_bins = np.linspace(M_range[0], M_range[1], bins)
    bin_size = magnitude_bins[1] - magnitude_bins[0]
    bin_center = magnitude_bins[:-1] + bin_size/2

    peak_median = np.zeros(len(bin_center))

    for ii in range(bins-1):
        peak_median[ii] = np.nanmedian(peak[(magnitude>=magnitude_bins[ii]) & (magnitude<magnitude_bins[ii+1])])
        distance_median = np.nanmedian(distance[(magnitude>=magnitude_bins[ii]) & (magnitude<magnitude_bins[ii+1])])
        peak_median[ii] = peak_median[ii]/distance_median**b_D
    return bin_center[~np.isnan(peak_median)], peak_median[~np.isnan(peak_median)]

fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
ax = ax.flatten()
for ii_region, region_key in enumerate(region_keys):
    temp_now = peak_amplitude_df_temp[peak_amplitude_df_temp.region==region_key]

    temp_now = temp_now.drop(index=temp_now[(temp_now.peak_P>=1e3)|(temp_now.peak_S>=1e3)].index)
    alpha=1

    bin_center, peak_P_median = running_median_scaled_by_distance(temp_now.magnitude, temp_now.peak_P, temp_now.distance_in_km, -1.269, bins=22)
    ax[0].semilogy(bin_center, peak_P_median, '-o', color=color_list[ii_region], label=region_title[ii_region], markersize=8, alpha=alpha)
    ax[0].vlines(x=2, ymin=1e-1, ymax=1e3, linestyle='--', color='k')

    bin_center, peak_S_median = running_median_scaled_by_distance(temp_now.magnitude, temp_now.peak_S, temp_now.distance_in_km, -1.588, bins=22)
    ax[1].semilogy(bin_center, peak_S_median, '-o', color=color_list[ii_region], label=region_title[ii_region], markersize=8, alpha=alpha)
    ax[1].vlines(x=2, ymin=1e-1, ymax=1e5, linestyle='--', color='k')


ax[0].set_title('P wave')
ax[1].legend(fontsize=10, loc=4)
ax[0].set_xlabel('Magnitude')
ax[0].set_ylabel('Median peak amplitude \nscaled by distance')
ax[1].set_title('S wave')
ax[1].set_xlabel('Magnitude')

ax = add_annotate(ax)
plt.savefig(figure_output_dir + '/magnitude_vs_mean_peak_amplitude_distance_scaled.png', bbox_inches='tight')


#%% show magnitude varying with mean peak amplitude for all events
# show magnitude varying with mean peak amplitude for all events
region_keys = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']
region_title = ['Ridgecrest', 'Long Valley N', 'Long Valley S', 'Sanriku']
color_list = ['#7DC0A6', '#ED926B', '#91A0C7', '#DA8EC0']

# check the good picked events
import glob
temp = glob.glob('/kuafu/jxli/Data/DASEventData/mammoth_north/temp_good/*.h5')
good_event_id_north = [int(temp1[-11:-3]) for temp1 in temp]
temp = glob.glob('/kuafu/jxli/Data/DASEventData/mammoth_south/temp_good/*.h5')
good_event_id_south = [int(temp1[-11:-3]) for temp1 in temp]


def running_median(magnitude, peak, bins=20, M_range=(0, 8)):
    magnitude = np.array(magnitude)
    peak = np.array(peak)

    magnitude_bins = np.linspace(M_range[0], M_range[1], bins)
    bin_size = magnitude_bins[1] - magnitude_bins[0]
    bin_center = magnitude_bins[:-1] + bin_size/2

    peak_median = np.zeros(len(bin_center))

    for ii in range(bins-1):
        peak_median[ii] = np.nanmedian(peak[(magnitude>=magnitude_bins[ii]) & (magnitude<magnitude_bins[ii+1])])
    return bin_center[~np.isnan(peak_median)], peak_median[~np.isnan(peak_median)]

fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
ax = ax.flatten()
for ii_region, region_key in enumerate(region_keys):
    temp_now = peak_amplitude_df_temp[peak_amplitude_df_temp.region==region_key]

    temp_now = temp_now.drop(index=temp_now[(temp_now.peak_P>=1e3)|(temp_now.peak_S>=1e3)].index)

    if region_key == 'mammothN':
        temp_now_good = temp_now[temp_now.event_id.isin(good_event_id_north)]
        alpha=0.5
    elif region_key == 'mammothS':
        temp_now_good = temp_now[temp_now.event_id.isin(good_event_id_south)]
        alpha=0.5
    else:
        alpha=1

    bin_center, peak_P_median = running_median(temp_now.magnitude, temp_now.peak_P, bins=22)
    ax[0].semilogy(bin_center, peak_P_median, '-o', color=color_list[ii_region], label=region_title[ii_region], markersize=8, alpha=alpha)
    ax[0].vlines(x=2, ymin=1e-2, ymax=10, linestyle='--', color='k')
    if 'mammoth' in region_key:
        bin_center_good, peak_P_median_good = running_median(temp_now_good.magnitude, temp_now_good.peak_P, bins=22)
        ax[0].semilogy(bin_center_good, peak_P_median_good, 'p', color=color_list[ii_region], label=region_title[ii_region]+', manual checked', markersize=12, markeredgecolor='k', zorder=-1)
    
    bin_center, peak_S_median = running_median(temp_now.magnitude, temp_now.peak_S, bins=22)
    ax[1].semilogy(bin_center, peak_S_median, '-o', color=color_list[ii_region], label=region_title[ii_region], markersize=8, alpha=alpha)
    ax[1].vlines(x=2, ymin=1e-2, ymax=50, linestyle='--', color='k')
    if 'mammoth' in region_key:
        bin_center_good, peak_S_median_good = running_median(temp_now_good.magnitude, temp_now_good.peak_S, bins=22)
        ax[1].semilogy(bin_center_good, peak_S_median_good, 'p', color=color_list[ii_region], label=region_title[ii_region]+', manual checked', markersize=12, markeredgecolor='k', zorder=-1)
    

ax[0].set_title('P wave')
ax[1].legend(fontsize=10, loc=4)
ax[0].set_xlabel('Magnitude')
ax[0].set_ylabel('Median peak amplitude')
ax[1].set_title('S wave')
ax[1].set_xlabel('Magnitude')

ax = add_annotate(ax)
plt.savefig(figure_output_dir + '/magnitude_vs_mean_peak_amplitude_manual_pick.png', bbox_inches='tight')

#%% show magnitude varying with mean peak amplitude for all events, SCALED BY DISTANCE
# show magnitude varying with mean peak amplitude for all events
region_keys = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']
region_title = ['Ridgecrest', 'Long Valley N', 'Long Valley S', 'Sanriku']
color_list = ['#7DC0A6', '#ED926B', '#91A0C7', '#DA8EC0']

# check the good picked events
import glob
temp = glob.glob('/kuafu/jxli/Data/DASEventData/mammoth_north/temp_good/*.h5')
good_event_id_north = [int(temp1[-11:-3]) for temp1 in temp]
temp = glob.glob('/kuafu/jxli/Data/DASEventData/mammoth_south/temp_good/*.h5')
good_event_id_south = [int(temp1[-11:-3]) for temp1 in temp]


def running_median_scaled_by_distance(magnitude, peak, distance, b_D, bins=20, M_range=(0, 8)):
    magnitude = np.array(magnitude)
    peak = np.array(peak)
    distance = np.array(distance)

    magnitude_bins = np.linspace(M_range[0], M_range[1], bins)
    bin_size = magnitude_bins[1] - magnitude_bins[0]
    bin_center = magnitude_bins[:-1] + bin_size/2

    peak_median = np.zeros(len(bin_center))

    for ii in range(bins-1):
        peak_median[ii] = np.nanmedian(peak[(magnitude>=magnitude_bins[ii]) & (magnitude<magnitude_bins[ii+1])])
        distance_median = np.nanmedian(distance[(magnitude>=magnitude_bins[ii]) & (magnitude<magnitude_bins[ii+1])])
        peak_median[ii] = peak_median[ii]/distance_median**b_D
    return bin_center[~np.isnan(peak_median)], peak_median[~np.isnan(peak_median)]

fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
ax = ax.flatten()
for ii_region, region_key in enumerate(region_keys):
    temp_now = peak_amplitude_df_temp[peak_amplitude_df_temp.region==region_key]

    temp_now = temp_now.drop(index=temp_now[(temp_now.peak_P>=1e3)|(temp_now.peak_S>=1e3)].index)

    if region_key == 'mammothN':
        temp_now_good = temp_now[temp_now.event_id.isin(good_event_id_north)]
        alpha=0.5
    elif region_key == 'mammothS':
        temp_now_good = temp_now[temp_now.event_id.isin(good_event_id_south)]
        alpha=0.5
    else:
        alpha=1

    bin_center, peak_P_median = running_median_scaled_by_distance(temp_now.magnitude, temp_now.peak_P, temp_now.distance_in_km, -1.269, bins=22)
    ax[0].semilogy(bin_center, peak_P_median, '-o', color=color_list[ii_region], label=region_title[ii_region], markersize=8, alpha=alpha)
    ax[0].vlines(x=2, ymin=1e-1, ymax=1e3, linestyle='--', color='k')
    if 'mammoth' in region_key:
        bin_center_good, peak_P_median_good = running_median_scaled_by_distance(temp_now_good.magnitude, temp_now_good.peak_P, temp_now_good.distance_in_km, -1.269, bins=22)
        ax[0].semilogy(bin_center_good, peak_P_median_good, 'p', color=color_list[ii_region], label=region_title[ii_region]+', manual checked', markersize=12, markeredgecolor='k', zorder=-1)
    
    bin_center, peak_S_median = running_median_scaled_by_distance(temp_now.magnitude, temp_now.peak_S, temp_now.distance_in_km, -1.588, bins=22)
    ax[1].semilogy(bin_center, peak_S_median, '-o', color=color_list[ii_region], label=region_title[ii_region], markersize=8, alpha=alpha)
    ax[1].vlines(x=2, ymin=1e-1, ymax=1e5, linestyle='--', color='k')
    if 'mammoth' in region_key:
        bin_center_good, peak_S_median_good = running_median_scaled_by_distance(temp_now_good.magnitude, temp_now_good.peak_S, temp_now_good.distance_in_km, -1.588, bins=22)
        ax[1].semilogy(bin_center_good, peak_S_median_good, 'p', color=color_list[ii_region], label=region_title[ii_region]+', manual checked', markersize=12, markeredgecolor='k', zorder=-1)
    

ax[0].set_title('P wave')
ax[1].legend(fontsize=10, loc=4)
ax[0].set_xlabel('Magnitude')
ax[0].set_ylabel('Median peak amplitude \nscaled by distance')
ax[1].set_title('S wave')
ax[1].set_xlabel('Magnitude')

ax = add_annotate(ax)
plt.savefig(figure_output_dir + '/magnitude_vs_mean_peak_amplitude_distance_scaled_manual_pick.png', bbox_inches='tight')

# %% Show the detectable figure after all quality control (M>2, SNR>5, min_channel>100)
# Show the detectable figure after all quality control (M>2, SNR>5, min_channel>100)

# apply event filter (directly load the filtered events)
sanriku_results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'

snr_threshold = 10
min_channel = 100
magnitude_threshold = [2, 10]
snr_threshold_Sanriku = 5

peak_amplitude_df_all = filter_event(peak_amplitude_df_all, M_threshold=magnitude_threshold, snr_threshold=5, min_channel=100)

# plot the M-D distribution for each region
peak_amplitude_df_temp = peak_amplitude_df_all.iloc[::1, :]
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
# peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P.astype('float'))
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float'))
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S
peak_amplitude_df_temp = peak_amplitude_df_temp.groupby(by=['event_id', 'region'], as_index=False).mean()

fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
fig.suptitle("Events with more than 100 available channels, SNR and M (>=2) filtered", fontsize=20)
ax = ax.flatten()
fig, ax = plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax)
plt.savefig(figure_output_dir + '/data_correlation_detectable_range_one_Magnitude_filtered_data.png', bbox_inches='tight')
#plt.savefig('/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/data_correlation_clipping_one.png', bbox_inches='tight')

if not show_region:
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle("Events with more than 100 available channels, SNR and M >=2 filtered", fontsize=20)
    ax = ax.flatten()
    fig, ax = plot_data_correlation(peak_amplitude_df_temp, M_clipping, D_clipping_P, D_clipping_S, D_sense_P, D_sense_S, ax, 
                                    show_region=False, show_clip=False)
    plt.savefig(figure_output_dir + '/data_correlation_detectable_range_one_Magnitude_filtered_data_present.png', bbox_inches='tight')
    #plt.savefig('/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/data_correlation_clipping_one.png', bbox_inches='tight')

# %%
# check the incorrect association
import glob
temp = glob.glob('/kuafu/jxli/Data/DASEventData/mammoth_north/temp_good/*.h5')
good_event_id_north = [int(temp1[-11:-3]) for temp1 in temp]

temp = glob.glob('/kuafu/jxli/Data/DASEventData/mammoth_south/temp_good/*.h5')
good_event_id_south = [int(temp1[-11:-3]) for temp1 in temp]

catalog_north = pd.read_csv('/kuafu/EventData/Mammoth_north/catalog.csv')
catalog_south = pd.read_csv('/kuafu/EventData/Mammoth_south/catalog.csv')
catalog_sanriku = pd.read_csv('/kuafu/EventData/Sanriku_ERI/catalog.csv')

good_event_north = catalog_north[catalog_north.event_id.isin(good_event_id_north)]
good_event_south = catalog_north[catalog_south.event_id.isin(good_event_id_south)]

event_north_all = catalog_north[catalog_north.event_id.isin(peak_amplitude_df_all[peak_amplitude_df_all.region=='mammothN'].event_id.unique())]
event_south_all = catalog_south[catalog_south.event_id.isin(peak_amplitude_df_all[peak_amplitude_df_all.region=='mammothS'].event_id.unique())]
event_sanriku_all = catalog_sanriku[catalog_sanriku.event_id.isin(peak_amplitude_df_all[peak_amplitude_df_all.region=='sanriku'].event_id.unique())]
#%%
t_diff = 10
from obspy import UTCDateTime

event_time_north = [obspy.UTCDateTime(parser.parse(time)) for time in good_event_north.event_time]
event_time_north_all = [obspy.UTCDateTime(parser.parse(time)) for time in event_north_all.event_time]
event_time_south = [obspy.UTCDateTime(parser.parse(time)) for time in good_event_south.event_time]
event_time_south_all = [obspy.UTCDateTime(parser.parse(time)) for time in event_south_all.event_time]
event_time_sanriku = [obspy.UTCDateTime(parser.parse(time)) for time in event_sanriku_all.event_time]


event_time_north = np.array([time-event_time_north[0] for time in event_time_north])
event_time_north_all = np.array([time-event_time_north_all[0] for time in event_time_north_all])
event_time_south = np.array([time-event_time_south[0] for time in event_time_south])
event_time_south_all = np.array([time-event_time_south_all[0] for time in event_time_south_all])
event_time_sanriku = np.array([time-event_time_sanriku[0] for time in event_time_sanriku])

print(f'North {t_diff}s')
event_time_north_days_interval = np.diff(event_time_north)
print(len(event_time_north_days_interval[event_time_north_days_interval<=t_diff]))
event_time_north_all_days_interval = np.diff(event_time_north_all)
print(len(event_time_north_all_days_interval[event_time_north_all_days_interval<=t_diff]))

print(f'South {t_diff}s')
event_time_south_days_interval = np.diff(event_time_south)
print(len(event_time_south_days_interval[event_time_south_days_interval<=t_diff]))
event_time_south_all_days_interval = np.diff(event_time_south_all)
print(len(event_time_south_all_days_interval[event_time_south_all_days_interval<=t_diff]))

print(f'Sanriku {t_diff}s')
event_time_sanriku_interval = np.diff(event_time_sanriku)
print(len(event_time_sanriku_interval[event_time_sanriku_interval<=t_diff]))
# %%
