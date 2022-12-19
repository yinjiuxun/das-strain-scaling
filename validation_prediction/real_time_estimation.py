#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

import sys
sys.path.append('../')
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
    'savefig.facecolor': 'white'
}
mpl.rcParams.update(params)

#%% 
# 1. Specify earthquake to look at
# load event waveforms
region_label = 'curie' #'ridgecrest' #'LA-Google' #'mammothN' #'ridgecrest'

weighted = 'ols' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise

time_step = 10 # time step to measure peaks

if 'ridgecrest' in region_label:
    event_folder = '/kuafu/EventData/Ridgecrest' 
    tt_dir = event_folder +  '/theoretical_arrival_time' 
    test_event_id = 40063391 #40063391(M4.57, New event, tt correction 0) 39493944(M5.8, tt correction 1.)  38548295(M4.90, tt correction 4.8), 38996632 (M4.89)
    tt_shift_p, tt_shift_s = 0, 0
    given_range_P = None
    given_range_S = None

if 'Google' in region_label:
    event_folder = '/kuafu/EventData/LA_Google' 
    tt_dir = event_folder +  '/theoretical_arrival_time' 

    test_event_id= 40241984 
    tt_shift_p, tt_shift_s = 0, 0 # To correct the time drift for Google data manually
    given_range_P = None
    given_range_S = None

    #test_event_id, given_range_S, ymin, ymax = 40241984, None, 50, 70 #40241984(2.88 from offshore) 
    # Another one from NW
    # test_event_id, given_range_P, given_range_S, ymin, ymax = 39914199, None, None, 40, 80 # (Another one M3 from NW)
    # test_event_id, given_range_P, given_range_S, ymin, ymax = 39974975, None, [40, 46], 35, 55 # (One M2.6 from N)
    #test_event_id, given_range_S, ymin, ymax = 40182560, [55, 60], 40, 90 #40182560(3.86)
    #test_event_id, given_range_S, ymin, ymax = 39970455, None, 55, 70 #39970455 (M2.79 from SW offshore)
    # test_event_id, given_range_S, ymin, ymax = 40296896, [0, 60], 0, 90 #40182560(2.36 from offshore) 
    #test_event_id, given_range_S, ymin, ymax = 39988031, [32, 38], 30, 45 # (2.09, View Park-Windsor Hills)
    #test_event_id, given_range_S, ymin, ymax = 39929895, None, 30, 50 # (3.26, South Gate)
    #test_event_id, given_range_S, ymin, ymax = 40180104, [30, 40], 30, 50 # (2.59, Hollywood)
    #test_event_id, tt_shift_p, tt_shift_s = 40194736, 0, 0 #40194736 (M3.98) a "large event" coming from NW of LA
    # test_event_id, given_range_S, ymin, ymax = 40194848, [55, 59], 40, 90 #40194848(3.05)
    #test_event_id, given_range_S, ymin, ymax = 40287104, [55, 60], 40, 90 #40287104(2.84)
    # test_event_id, given_range_P, given_range_S, ymin, ymax = 39974135, [40, 50], [50, 60], 40, 80 # (3.28, Ontario)
    #test_event_id, tt_shift_p, tt_shift_s = 39882375, 0, 0 #39882375 (M3.6)
    #test_event_id, tt_shift_p, tt_shift_s = 40033927, 0, 0 #39882375 (M1.53)
    #test_event_id, tt_shift_p, tt_shift_s = 39983383, 0, 0 #39983383 (M2.32)  39983383 is an event below the DAS array
    #40182560 (M3.8)
    #test_event_id, tt_shift_p, tt_shift_s = 40019919, 0, 0 #40019919 (M3.76)
    #test_event_id, tt_shift_p, tt_shift_s = 39929895, 0, 0 #39929895 (M3.26)
    #test_event_id, tt_shift_p, tt_shift_s = 40033063, 3.9, 5.1 # 40033063 is an event below the DAS array

elif 'mammoth' in region_label:
    event_folder = '/kuafu/EventData/Mammoth_north'#'/kuafu/EventData/Mammoth_south' #'/kuafu/EventData/Ridgecrest'
    event_folder = '/kuafu/EventData/AlumRock5.1/MammothNorth'
    tt_dir = event_folder +  '/model_proc_tt/CVM3D' ##
    test_event_id = 73799091 # 73584926(M6) 73481241(M5) 73585021(M4.6)
    tt_shift_p, tt_shift_s = -3, -6.5
    given_range_P = None
    given_range_S = None

elif 'arcata' in region_label:
    event_folder = '/kuafu/EventData/Arcata_Spring2022'#'/kuafu/EventData/Mammoth_south' #'/kuafu/EventData/Ridgecrest'
    tt_dir = event_folder +  '/theoretical_arrival_time_calibrated' ##
    test_event_id =  73743421 # 73743421 (3.79), 73739346(3.64) 73736021(3.05)
    tt_shift_p, tt_shift_s = 0, 0
    given_range_P = None
    given_range_S = None

elif 'curie' in region_label:
    event_folder = '/kuafu/EventData/Curie'#'/kuafu/EventData/Mammoth_south' #'/kuafu/EventData/Ridgecrest'
    tt_dir = event_folder +  '/theoretical_arrival_time_calibrated' ##
    test_event_id =  9001 #
    tt_shift_p, tt_shift_s = 0, 0
    given_range_P = None
    given_range_S = None

catalog = pd.read_csv(event_folder + '/catalog.csv')
das_waveform_path = event_folder + '/data'

#%%
# 2. Specify the array to look at
# load the DAS array information
DAS_info = pd.read_csv(event_folder + '/das_info.csv')
# DAS_info = pd.read_csv('/kuafu/EventData/Mammoth_south/das_info.csv')
# specify the directory of ML picking
ML_picking_dir = event_folder + '/picks_phasenet_das'
use_ML_picking = False
if use_ML_picking:
    picking_label='ml'
else:
    picking_label='vm'

#%%
# 3. Specify the coefficients to load
# regression coefficients of the multiple array case
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/'
regression_dir = f'iter_regression_results_smf{weight_text}_100_channel_at_least'


# # results of individual array
# results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate/'
# regression_dir = f'iter_regression_results_smf_weighted_100_channel_at_least'

# results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/'
# regression_dir = f'iter_regression_results_smf_weighted_100_channel_at_least'
#%%
# make figure output directory
fig_output_dir = results_output_dir + '/' + regression_dir + '/estimated_M'
if not os.path.exists(fig_output_dir):
    os.mkdir(fig_output_dir)

#%%
# Load DAS channels
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['index']
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

# Load the DAS data
strain_rate, info = load_event_data(das_waveform_path, test_event_id)
if 'ridgecrest' in region_label:
    strain_rate = strain_rate#[:, DAS_index]
else:
    strain_rate = strain_rate[:, DAS_index]
das_dt = info['dt_s']
nt = strain_rate.shape[0]

if 'curie' in region_label:
    das_time0 = np.arange(nt) * das_dt
else:
    das_time0 = np.arange(nt) * das_dt - 30

# Load regression results
regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_combined_site_terms_iter.pickle")
regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_iter.pickle")
site_terms_df = pd.read_csv(results_output_dir + '/' + regression_dir + f"/site_terms_iter.csv")

if 'arcata' in region_label: # actually, if transfer scaling
    results_dir = f'/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/transfer_regression_specified_smf{weight_text}_100_channel_at_least/'
    #results_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/transfer_regression_test_smf_weighted_100_channel_at_least/3_fit_events_6th_test'
    site_terms_df = pd.read_csv(results_dir + '/site_terms_transfer.csv')
    DAS_channel_num = DAS_info['index'].max()
if 'curie' in region_label: # actually, if transfer scaling
    results_dir = f'/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate/transfer_regression_specified_smf{weight_text}_100_channel_at_least_9007/'
    #results_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/transfer_regression_test_smf_weighted_100_channel_at_least/3_fit_events_6th_test'
    site_terms_df = pd.read_csv(results_dir + '/site_terms_transfer.csv')
    DAS_channel_num = len(DAS_info)
if 'Google' in region_label: # actually, if transfer scaling
    LA_results_dir = f'/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/transfer_regression_test_smf{weight_text}_100_channel_at_least/9_fit_events_4th_test/'
    site_terms_df = pd.read_csv(LA_results_dir + '/site_terms_transfer.csv')
    LA_results_dir = f'/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/iter_regression_results_smf{weight_text}_100_channel_at_least'
    site_terms_df = pd.read_csv(LA_results_dir + '/site_terms_iter.csv')
#%%
# have the site term in the same shape of original data, fill nan instead
channel_id = np.array(site_terms_df[site_terms_df.region == region_label]['channel_id'].astype('int'))
site_terms_P = np.zeros(shape=(1, DAS_channel_num)) * np.nan
site_terms_S = np.zeros(shape=(1, DAS_channel_num)) * np.nan

site_terms_P[:, channel_id] = site_terms_df[site_terms_df.region == region_label]['site_term_P']
site_terms_S[:, channel_id] = site_terms_df[site_terms_df.region == region_label]['site_term_S']

if 'ridgecrest' in region_label:
    pass
elif 'mammoth' in region_label:
    pass
else:
    site_terms_P = site_terms_P[:, np.array(DAS_index)-1]
    site_terms_S = site_terms_S[:, np.array(DAS_index)-1]
#%%
# load the travel time and process
def remove_ml_tt_outliers(ML_picking, das_dt, tdiff=10, given_range=None):
    temp = ML_picking.drop(index=ML_picking[abs(ML_picking.phase_index - ML_picking.phase_index.median())*das_dt >= tdiff].index)
    if given_range:
        try:
            temp = temp[(temp.phase_index>=given_range[0]/das_dt) & (temp.phase_index<=given_range[1]/das_dt)]
        except:
            print('cannot specify range, skip...')
    return temp

eq_info = catalog[catalog.event_id == test_event_id]
eq_lat = eq_info.latitude # lat
eq_lon = eq_info.longitude # lon
eq_mag = eq_info.magnitude # catalog magnitude
eq_time = eq_info.event_time # eq time


# get distance from earthquake to each channel
distance_to_source = locations2degrees(DAS_lat, DAS_lon, eq_lat, eq_lon) * 113 # in km
distance_to_source = np.sqrt(eq_info.iloc[0, :].depth_km**2 + distance_to_source**2)
distance_to_source = distance_to_source[np.newaxis, :]

# First try ML picking if specified, or turn to theoretical TT
if use_ML_picking: #TODO: check why there is an obvious difference of picking
    tt_tp = np.zeros(shape=(1, DAS_channel_num))*np.nan
    tt_ts = tt_tp.copy()
    try:
        ML_picking = pd.read_csv(ML_picking_dir + f'/{test_event_id}.csv')
        if 'mammoth' in region_label:
            ML_picking = ML_picking[ML_picking.channel_index.isin(DAS_index)]

        ML_picking_P = ML_picking[ML_picking.phase_type == 'P']
        ML_picking_S = ML_picking[ML_picking.phase_type == 'S']
        ML_picking_P = remove_ml_tt_outliers(ML_picking_P, das_dt, tdiff=5, given_range=given_range_P)
        ML_picking_S = remove_ml_tt_outliers(ML_picking_S, das_dt, tdiff=20, given_range=given_range_S)

        tt_tp[0, ML_picking_P.channel_index] = das_time0[ML_picking_P.phase_index]
        tt_ts[0, ML_picking_S.channel_index] = das_time0[ML_picking_S.phase_index]

    except:
        print("didn't find ML picking, use theoretical tt instead...")
        use_ML_picking = False
        picking_label = 'vm'

    if 'arcata' in region_label:
        tt_tp = tt_tp[:, np.array(DAS_index)-1]
        tt_tp = tt_tp[:, np.array(DAS_index)-1]

if not use_ML_picking:            
    # theoretical travel time 
    if 'mammoth' in region_label:
        cvm_tt = pd.read_csv(tt_dir + f'/{test_event_id}.csv')
        tt_tp = np.array(cvm_tt.tp)
        tt_tp = tt_tp[np.newaxis, :]
        tt_ts = np.array(cvm_tt.ts)
        tt_ts = tt_ts[np.newaxis, :]

        # For travel time from velocity model, may need some manual correction
        tt_tp= tt_tp+tt_shift_p 
        tt_ts= tt_ts+tt_shift_s

    elif ('ridgecrest' in region_label):
        cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
        tt_tp = np.array(cvm_tt.P_arrival)-30
        tt_tp = tt_tp[np.newaxis, :]
        tt_ts = np.array(cvm_tt.S_arrival)-30
        tt_ts = tt_ts[np.newaxis, :]

        # For travel time from velocity model, may need some manual correction
        tt_tp= tt_tp+tt_shift_p 
        tt_ts= tt_ts+tt_shift_s

    elif ('arcata' in region_label):
        cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
        tt_tp = np.array(cvm_tt.P_arrival)-30
        tt_tp = tt_tp[np.newaxis, :]
        tt_ts = np.array(cvm_tt.S_arrival)-30
        tt_ts = tt_ts[np.newaxis, :]

        # For travel time from velocity model, may need some manual correction
        tt_tp= tt_tp+tt_shift_p 
        tt_ts= tt_ts+tt_shift_s

    elif ('curie' in region_label):
        cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
        tt_tp = np.array(cvm_tt.P_arrival)
        tt_tp = tt_tp[np.newaxis, :]
        tt_ts = np.array(cvm_tt.S_arrival)
        tt_ts = tt_ts[np.newaxis, :]

        # For travel time from velocity model, may need some manual correction
        tt_tp= tt_tp+tt_shift_p 
        tt_ts= tt_ts+tt_shift_s

    elif 'Google' in region_label:
        cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
        tt_tp = np.array(cvm_tt.P_arrival)
        tt_tp = tt_tp[np.newaxis, :]
        tt_ts = np.array(cvm_tt.S_arrival)
        tt_ts = tt_ts[np.newaxis, :]
        # For Google data, there may be some time drift that needs to be corrected
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
# mag_estimate_final[das_time_mat>=cvm_tt_ts] = mag_estimate_S[das_time_mat>=cvm_tt_ts]
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

fig, ax = plt.subplots(3, 1, figsize=(10,20))
# Strain
pclip=99
clipVal = np.percentile(np.absolute(strain_rate), pclip)
gca = ax[0]
clb = gca.imshow(strain_rate.T, 
            extent=[das_time0[0], das_time0[-1], mag_estimate_P.shape[1], 0],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'), interpolation='none')

# if use_ML_picking:
#     gca.plot(tt_tp[0, ML_picking_P.channel_index], ML_picking_P.channel_index, '-k', linewidth=4)
#     gca.plot(tt_ts[0, ML_picking_S.channel_index], ML_picking_S.channel_index, '-k', linewidth=4)
# else:
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
gca.set_ylabel('Estimated Magnitude')
gca.legend(loc=4)

plt.subplots_adjust(hspace=0.2)

# some final handling
letter_list = [str(chr(k+97)) for k in range(0, 3)]
k=0
for ii in range(0, 3):
    ax[ii].annotate(f'({letter_list[k]})', xy=(-0.05, 1.0), xycoords=ax[ii].transAxes)
    k+=1

plt.savefig(fig_output_dir + f'/{region_label}_{test_event_id}_estimated_mag_image_{picking_label}.png', bbox_inches='tight')
plt.savefig(fig_output_dir + f'/{region_label}_{test_event_id}_estimated_mag_image_{picking_label}.pdf', bbox_inches='tight')
# fig.tight_layout()


# %%
