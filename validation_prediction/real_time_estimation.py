#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.loading import load_event_data

from obspy.geodetics import locations2degrees
import matplotlib as mpl
import matplotlib.pyplot as plt

# set plotting parameters 
fontsize=18
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 100,  # to adjust notebook inline plot size
    'axes.labelsize': fontsize, # fontsize for x and y labels (was 10)
    'axes.titlesize': fontsize,
    'font.size': fontsize,
    'legend.fontsize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'text.usetex':False,
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42 # Turn off text conversion to outlines
}
mpl.rcParams.update(params)

#%% 
# 1. Specify earthquake to look at
# load event waveforms
region_label = 'mammothN' #'ridgecrest' 

weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise

time_step = 10 # time step to measure peaks

if 'ridgecrest' in region_label:
    event_folder = '../data_files/event_data' 
    catalog = pd.read_csv('../data_files/catalogs/catalog_Ridgecrest.csv')
    DAS_info = pd.read_csv('../data_files/das_info/das_info_Ridgecrest.csv')

    tt_dir = event_folder
    test_event_id = 40063391 
    tt_shift_p, tt_shift_s = 0, 0


elif 'mammoth' in region_label:
    event_folder = '../data_files/event_data' 
    catalog = pd.read_csv('../data_files/catalogs/catalog_LongValley_N.csv')
    DAS_info = pd.read_csv('../data_files/das_info/das_info_LongValley_N.csv')

    tt_dir = event_folder
    test_event_id = 73799091 
    tt_shift_p, tt_shift_s = -3, -6.5

#%%
# 3. Specify the coefficients to load
# regression coefficients of the multiple array case
results_output_dir = '../iter_results'

# make figure output directory
fig_output_dir = results_output_dir + '/estimated_M'
mkdir(fig_output_dir)

#%%
# Load DAS channels
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['index']
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

# Load the DAS data
strain_rate, info = load_event_data(event_folder, test_event_id)
if 'ridgecrest' in region_label:
    strain_rate = strain_rate#[:, DAS_index]
else:
    strain_rate = strain_rate[:, DAS_index]
das_dt = info['dt_s']
nt = strain_rate.shape[0]
das_time0 = np.arange(nt) * das_dt-30

# Load regression results
regP = sm.load(results_output_dir + f"/P_regression_combined_site_terms_iter.pickle")
regS = sm.load(results_output_dir + f"/S_regression_combined_site_terms_iter.pickle")
site_terms_df = pd.read_csv(results_output_dir + f"/site_terms_iter.csv")

#%%
# have the site term in the same shape of original data, fill nan instead
site_terms_P = np.zeros(shape=(1, DAS_channel_num)) * np.nan
site_terms_S = np.zeros(shape=(1, DAS_channel_num)) * np.nan

site_terms_df_P_temp = site_terms_df[(site_terms_df.region == region_label) & (site_terms_df.wavetype == 'P')]
site_terms_df_S_temp = site_terms_df[(site_terms_df.region == region_label) & (site_terms_df.wavetype == 'S')]

site_terms_P[:, site_terms_df_P_temp['channel_id'].astype('int')] = site_terms_df_P_temp['site_term_P']
site_terms_S[:, site_terms_df_S_temp['channel_id'].astype('int')] = site_terms_df_S_temp['site_term_S']

if 'ridgecrest' in region_label:
    pass
elif 'mammoth' in region_label:
    pass
else:
    site_terms_P = site_terms_P[:, np.array(DAS_index)-1]
    site_terms_S = site_terms_S[:, np.array(DAS_index)-1]
#%%
# # load the travel time and process
eq_info = catalog[catalog.event_id == test_event_id]
eq_lat = eq_info.latitude # lat
eq_lon = eq_info.longitude # lon
eq_mag = eq_info.magnitude # catalog magnitude
eq_time = eq_info.event_time # eq time


# get distance from earthquake to each channel
distance_to_source = locations2degrees(DAS_lat, DAS_lon, eq_lat, eq_lon) * 113 # in km
distance_to_source = np.sqrt(eq_info.iloc[0, :].depth_km**2 + distance_to_source**2)
distance_to_source = distance_to_source[np.newaxis, :]

cvm_tt = pd.read_csv(tt_dir + f'/tt_{test_event_id}.csv')
tt_tp = np.array(cvm_tt.P_arrival)
tt_tp = tt_tp[np.newaxis, :]
tt_ts = np.array(cvm_tt.S_arrival)
tt_ts = tt_ts[np.newaxis, :]

# For travel time from velocity model, may need some manual correction
tt_tp= tt_tp+tt_shift_p 
tt_ts= tt_ts+tt_shift_s

# Some DUMP process to handle the arrival time
tt_tp_temp = tt_tp.copy()
tt_tp_temp[np.isnan(tt_tp_temp)]=1e10
tt_ts_temp = tt_ts.copy()
tt_ts_temp[np.isnan(tt_ts_temp)]=1e10

#%%
#extract peak amplitude based on the phase arrival time
strain_rate_clipped_P = strain_rate.copy()
das_time1 = das_time0[:, np.newaxis]
strain_rate_clipped_P[das_time1<=tt_tp_temp]=np.nan
strain_rate_clipped_P[(das_time1>tt_tp_temp+2)] = np.nan

strain_rate_clipped_S = strain_rate.copy()
strain_rate_clipped_S[(das_time1<tt_ts_temp)] = np.nan
strain_rate_clipped_S[(das_time1>tt_ts_temp+2)] = np.nan

data_peak_mat_P = np.zeros((np.ceil(strain_rate.shape[0]/time_step).astype('int'), strain_rate.shape[1]))
data_peak_mat_S = np.zeros((np.ceil(strain_rate.shape[0]/time_step).astype('int'), strain_rate.shape[1]))
for i, ii_win in enumerate(range(0, strain_rate.shape[0], time_step)):
    # keep the maximum unchanged
    data_peak_mat_P[i, :] = np.nanmax(abs(strain_rate_clipped_P[0:(ii_win+time_step), :]), axis=0)     
    data_peak_mat_S[i, :] = np.nanmax(abs(strain_rate_clipped_S[0:(ii_win+time_step), :]), axis=0)    

das_time = das_time0[::time_step]

#%%
# calculate magnitude for each channel
mag_estimate_P = (np.log10(data_peak_mat_P+1e-12) - site_terms_P - np.log10(distance_to_source)*regP.params['np.log10(distance_in_km)'])/regP.params['magnitude']
median_mag_P = np.nanmedian(mag_estimate_P, axis=1)
mean_mag_P = np.nanmean(mag_estimate_P, axis=1)
std_mag_P = np.nanstd(mag_estimate_P, axis=1)

mag_estimate_S = (np.log10(data_peak_mat_S+1e-12) - site_terms_S - np.log10(distance_to_source)*regS.params['np.log10(distance_in_km)'])/regS.params['magnitude']
median_mag_S = np.nanmedian(mag_estimate_S, axis=1)
mean_mag_S = np.nanmean(mag_estimate_S, axis=1)
std_mag_S = np.nanstd(mag_estimate_S, axis=1)

# combine both P and S to give final magnitude 
das_time_mat = das_time[:, np.newaxis]
mag_estimate_final = mag_estimate_P.copy()
mag_estimate_final = np.nanmedian(np.array([mag_estimate_P, mag_estimate_S]), axis=0)

median_mag = np.nanmedian(mag_estimate_final, axis=1)
mean_mag = np.nanmean(mag_estimate_final, axis=1)
std_mag = np.nanstd(mag_estimate_final, axis=1)

#%% 
# Having all together as one figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

time_rang_show = [0, 120]
if np.isnan(np.nanmin(tt_tp)):
    time_rang_show[0] = np.nanmin(tt_ts)-10
else:
    time_rang_show[0] = np.nanmin(tt_tp)-3
if not np.isnan(np.nanmax(tt_ts)):
    time_rang_show[1] = np.nanmedian(tt_ts)+10
    
if test_event_id == 40063391:
    time_rang_show = [1, 10]
if test_event_id == 73799091:
    time_rang_show = [30, 75]

mpl.rcParams.update(params)
fig, ax = plt.subplots(3, 1, figsize=(8,16))
# Strain
pclip=99
clipVal = np.percentile(np.absolute(strain_rate), pclip)
gca = ax[0]
clb = gca.imshow(strain_rate.T, 
            extent=[das_time0[0], das_time0[-1], mag_estimate_P.shape[1], 0],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'), interpolation='none')

gca.plot(tt_tp.flatten(), np.arange(tt_tp.shape[1]), '--k', linewidth=2, label='P')
gca.plot(tt_ts.flatten(), np.arange(tt_ts.shape[1]), '-k', linewidth=2, label='S')

gca.vlines(x=[np.min(tt_tp), np.min(tt_ts)], ymax=strain_rate.shape[1], ymin=0, label='earliest tP and tS', color='b')
gca.vlines(x=[np.max(tt_tp)+2, np.max(tt_ts)+2], ymax=strain_rate.shape[1], ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s')

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.invert_yaxis()

axins1 = inset_axes(gca,
                    width="2%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='lower left')
gca.set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}', fontsize=20)
fig.colorbar(clb, cax=axins1, orientation="vertical", label='strain rate ($10^{-6}$/s)')

# Colormap for Magnitude
cmap = plt.cm.get_cmap('OrRd', 6)

gca = ax[1]
clb = gca.imshow(mag_estimate_final.T, 
            extent=[das_time[0], das_time[-1], mag_estimate_final.shape[1], 0],
            aspect='auto', vmin=0, vmax=6, cmap=cmap, interpolation='none')
gca.plot(tt_tp.flatten(), np.arange(tt_tp.shape[1]), '-k', linewidth=4)
gca.plot(tt_ts.flatten(), np.arange(tt_ts.shape[1]), '-k', linewidth=4)
gca.vlines(x=[np.min(tt_tp), np.min(tt_ts)], ymax=mag_estimate_final.shape[1], ymin=0, label='earliest tP and tS', color='b', zorder=10)
gca.vlines(x=[np.max(tt_tp)+2, np.max(tt_ts)+2], ymax=mag_estimate_final.shape[1], ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s', zorder=10)

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.invert_yaxis()

axins2 = inset_axes(gca,
                    width="2%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='lower left')
fig.colorbar(clb, cax=axins2, orientation="vertical", label='Magnitude')

gca=ax[2]
# gca.plot(das_time, mag_estimate_final, '-k', linewidth=0.1, alpha=0.1)
gca.plot(das_time, median_mag, '-r', linewidth=4, alpha=1, zorder=3, label='mean')
gca.plot(das_time, median_mag-std_mag, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
gca.plot(das_time, median_mag+std_mag, '--r', linewidth=2, alpha=0.5, zorder=3)
gca.vlines(x=[np.nanmin(tt_tp), np.nanmin(tt_ts)], ymax=10, ymin=0, label='earliest tP and tS', color='b')
gca.vlines(x=[np.nanmax(tt_tp)+2, np.nanmax(tt_ts)+2], ymax=10, ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s')

mag_estiamte_P = median_mag[int(np.nanmax(tt_tp+2+30)/das_dt/time_step)]
gca.text(x=np.nanmax(tt_tp+2), y=mag_estiamte_P+0.3, s=f'M {mag_estiamte_P:.2f}', color='r', fontsize=18)
mag_estiamte_S = median_mag[int(np.nanmax(tt_ts+2+30)/das_dt/time_step)]
gca.text(x=np.nanmax(tt_ts+2), y=mag_estiamte_S+0.3, s=f'M {mag_estiamte_S:.2f}', color='r', fontsize=18)

gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')
gca.set_yticks(np.arange(0, 9))
gca.set_ylim(0, eq_mag.values*1.4)
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.set_xlabel('Time (s)')
gca.set_ylabel('Magnitude')
gca.legend(loc=4, fontsize=15)

plt.subplots_adjust(hspace=0.2)

# some final handling
letter_list = [str(chr(k+100)) for k in range(0, 3)]
k=0
for ii in range(0, 3):
    ax[ii].annotate(f'({letter_list[k]})', xy=(-0.1, 1.0), xycoords=ax[ii].transAxes)
    k+=1

plt.savefig(fig_output_dir + f'/{region_label}_{test_event_id}_estimated_mag.png', bbox_inches='tight')

# %%
