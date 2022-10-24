#%% import modules
from configparser import Interpolation
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

from utility.loading import load_event_data

from obspy.geodetics import locations2degrees
import matplotlib as mpl
import matplotlib.pyplot as plt

# set plotting parameters 
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 100,  # to adjust notebook inline plot size
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
mpl.rcParams.update(params)

# ========================== work on Combined results from RM ================================
# First check how well the regression relation can be used to calculate Magnitude
# Then try to use the regression relation from small events to predict the larger ones
#%% 
# load the results from combined regional site terms t
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/'#'/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RO'
regression_dir = f'iter_regression_results_smf_weighted_100_channel_at_least'

# load event waveforms
region_label = 'LA-Google' #'LA-Google' #'mammothN' #'ridgecrest'

if 'ridgecrest' in region_label:
    event_folder = '/kuafu/EventData/Ridgecrest' 
    tt_dir = event_folder +  '/theoretical_arrival_time' 
    test_event_id = 40063391 #40063391(M4.57) 39493944(M5.8) 

if 'Google' in region_label:
    event_folder = '/kuafu/EventData/LA_Google' 
    tt_dir = event_folder +  '/theoretical_arrival_time' 

    LA_results_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/transfer_regression_test_smf_weighted_100_channel_at_least/9_fit_events_4th_test/'
    temp =np.load(LA_results_dir + '/transfer_event_list.npz')
    event_id_fit=temp['event_id_fit']
    event_id_predict=temp['event_id_predict']

elif 'mammoth' in region_label:
    event_folder = '/kuafu/EventData/Mammoth_north'#'/kuafu/EventData/Mammoth_south' #'/kuafu/EventData/Ridgecrest'
    tt_dir = event_folder +  '/model_proc_tt/CVM3D' ##
    test_event_id = 73481241 # 73584926(M6) 73481241(M5) 73585021(M4.6)
    
# load catalog
catalog = pd.read_csv(event_folder + '/catalog.csv')
# load the DAS array information
DAS_info = pd.read_csv(event_folder + '/das_info.csv')
# specify the directory of ML picking
ML_picking_dir = event_folder + '/picks_phasenet_das'

DAS_channel_num = DAS_info.shape[0]
# DAS_index = DAS_info['index'].astype('int')
DAS_index = DAS_info.index
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

das_path = event_folder + '/data'

# only for LA Google
if 'Google' in region_label:
    event_to_look = catalog[catalog.event_id.isin(event_id_predict)]
    test_event_id, tt_shift_p, tt_shift_s = 40194736, 0, 0 #40194736 (M3.98) a "large event" coming from NW of LA
    #test_event_id, tt_shift_p, tt_shift_s = 40194736, 0, 0 #40194736 (M3.98) a "large event" coming from NW of LA
    #test_event_id, tt_shift_p, tt_shift_s = 39882375, 0, 0 #39882375 (M3.6)
    #test_event_id, tt_shift_p, tt_shift_s = 40033927, 0, 0 #39882375 (M1.53)
    #test_event_id, tt_shift_p, tt_shift_s = 39983383, 0, 0 #39983383 (M2.32)  39983383 is an event below the DAS array
    #40182560 (M3.8)
    #test_event_id, tt_shift_p, tt_shift_s = 40019919, 0, 0 #40019919 (M3.76)
    #test_event_id, tt_shift_p, tt_shift_s = 39929895, 0, 0 #39929895 (M3.26)
    #test_event_id, tt_shift_p, tt_shift_s = 40033063, 3.9, 5.1 # 40033063 is an event below the DAS array
#%% 
# make figure output directory
fig_output_dir = results_output_dir + '/' + regression_dir + '/estimated_M'
if not os.path.exists(fig_output_dir):
    os.mkdir(fig_output_dir)

#%% # ========================== Real time estimation ================================
# load regression parameters
regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_combined_site_terms_iter.pickle")
regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_iter.pickle")
site_terms_df = pd.read_csv(results_output_dir + '/' + regression_dir + f"/site_terms_iter.csv")

if 'Google' in region_label:
    site_terms_df = pd.read_csv(LA_results_dir + '/site_terms_transfer.csv')

# TODO: have the site term in the same shape of original data, fill nan instead
#site_terms = site_terms[DAS_index]
channel_id = np.array(site_terms_df[site_terms_df.region == region_label]['channel_id'].astype('int'))
site_terms_P = np.zeros(shape=(1, DAS_channel_num)) * np.nan
site_terms_S = np.zeros(shape=(1, DAS_channel_num)) * np.nan

site_terms_P[:, channel_id] = site_terms_df[site_terms_df.region == region_label]['site_term_P']
site_terms_S[:, channel_id] = site_terms_df[site_terms_df.region == region_label]['site_term_S']
#%%
def remove_ml_tt_outliers(ML_picking, das_dt, tdiff=10):
    return ML_picking.drop(index=ML_picking[abs(ML_picking.phase_index - ML_picking.phase_index.median())*das_dt >= tdiff].index)




# extract peak amplitude
eq_info = catalog[catalog.event_id == test_event_id]
eq_lat = eq_info.latitude # lat
eq_lon = eq_info.longitude # lon
eq_mag = eq_info.magnitude # catalog magnitude
eq_time = eq_info.event_time # eq time

# Load the DAS data
strain_rate, info = load_event_data(das_path, test_event_id)
strain_rate = strain_rate[:, DAS_index]
das_dt = info['dt_s']
nt = strain_rate.shape[0]
das_time0 = np.arange(nt) * das_dt

# get distance from earthquake to each channel
distance_to_source = locations2degrees(DAS_lat, DAS_lon, eq_lat, eq_lon) * 113 # in km
distance_to_source = distance_to_source[np.newaxis, :]

# theoretical travel time 
if 'mammoth' in region_label:
    cvm_tt = pd.read_csv(tt_dir + f'/{test_event_id}.csv')
    cvm_tt_tp = np.array(cvm_tt.tp)+30
    cvm_tt_tp = cvm_tt_tp[np.newaxis, :]
    cvm_tt_ts = np.array(cvm_tt.ts)+30
    cvm_tt_ts = cvm_tt_ts[np.newaxis, :]
elif ('ridgecrest' in region_label):
    cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
    cvm_tt_tp = np.array(cvm_tt.P_arrival)
    cvm_tt_tp = cvm_tt_tp[np.newaxis, :]
    cvm_tt_ts = np.array(cvm_tt.S_arrival)
    cvm_tt_ts = cvm_tt_ts[np.newaxis, :]

use_ML_picking = True
if 'Google' in region_label: # to correct for the clock drifting issue
    if use_ML_picking:
        cvm_tt_tp = np.zeros(shape=(1, DAS_channel_num))*np.nan
        cvm_tt_ts = cvm_tt_tp.copy()
        try:
            ML_picking = pd.read_csv(ML_picking_dir + f'/{test_event_id}.csv')
            ML_picking_P = ML_picking[ML_picking.phase_type == 'P']
            ML_picking_S = ML_picking[ML_picking.phase_type == 'S']
            ML_picking_P = remove_ml_tt_outliers(ML_picking_P, das_dt, tdiff=5)
            ML_picking_S = remove_ml_tt_outliers(ML_picking_S, das_dt, tdiff=5)

            cvm_tt_tp[0, ML_picking_P.channel_index] = das_time0[ML_picking_P.phase_index]
            cvm_tt_ts[0, ML_picking_S.channel_index] = das_time0[ML_picking_S.phase_index]


        except:
            print("didn't find ML picking, use theoretical tt instead...")
            use_ML_picking = False

    if not use_ML_picking:
        cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
        cvm_tt_tp = np.array(cvm_tt.P_arrival)
        cvm_tt_tp = cvm_tt_tp[np.newaxis, :]
        cvm_tt_ts = np.array(cvm_tt.S_arrival)
        cvm_tt_ts = cvm_tt_ts[np.newaxis, :]

        cvm_tt_tp= cvm_tt_tp+tt_shift_p
        cvm_tt_ts= cvm_tt_ts+tt_shift_s

cvm_tt_tp_temp = cvm_tt_tp.copy()
cvm_tt_tp_temp[np.isnan(cvm_tt_tp_temp)]=1e10
cvm_tt_ts_temp = cvm_tt_ts.copy()
cvm_tt_ts_temp[np.isnan(cvm_tt_ts_temp)]=1e10

strain_rate_clipped_P = strain_rate.copy()
das_time1 = das_time0[:, np.newaxis]
strain_rate_clipped_P[das_time1<=cvm_tt_tp_temp]=np.nan
strain_rate_clipped_P[(das_time1>cvm_tt_tp_temp+2)] = np.nan


strain_rate_clipped_S = strain_rate.copy()
strain_rate_clipped_S[(das_time1<cvm_tt_ts_temp)] = np.nan
strain_rate_clipped_S[(das_time1>cvm_tt_ts_temp+2)] = np.nan

time_step = 100
data_peak_mat_P = np.zeros((np.ceil(strain_rate.shape[0]/time_step).astype('int'), strain_rate.shape[1]))
data_peak_mat_S = np.zeros((np.ceil(strain_rate.shape[0]/time_step).astype('int'), strain_rate.shape[1]))
for i, ii_win in enumerate(range(0, strain_rate.shape[0], time_step)):
    # data_peak_mat[i, :] = np.nanmax(abs(strain_rate_clipped[ii_win:(ii_win+time_step), :]), axis=0)  

    # keep the maximum unchanged
    data_peak_mat_P[i, :] = np.nanmax(abs(strain_rate_clipped_P[0:(ii_win+time_step), :]), axis=0)     
    data_peak_mat_S[i, :] = np.nanmax(abs(strain_rate_clipped_S[0:(ii_win+time_step), :]), axis=0)    

das_time = das_time0[::time_step]

# # only keep available channels
# distance_to_source= distance_to_source[:, channel_id]
# data_peak_mat_P = data_peak_mat_P[:, channel_id]
# data_peak_mat_S = data_peak_mat_S[:, channel_id]

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
mag_estimate_final = np.nanmean(np.array([mag_estimate_P, mag_estimate_S]), axis=0)

median_mag = np.nanmedian(mag_estimate_final, axis=1)
mean_mag = np.nanmean(mag_estimate_final, axis=1)
std_mag = np.nanstd(mag_estimate_final, axis=1)


#%% Having all together as one figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
time_rang_show = [0, 120]
if np.isnan(np.nanmin(cvm_tt_tp)):
    time_rang_show[0] = np.nanmin(cvm_tt_ts)-10
else:
    time_rang_show[0] = np.nanmin(cvm_tt_tp)-3
if not np.isnan(np.nanmax(cvm_tt_ts)):
    time_rang_show[1] = np.nanmedian(cvm_tt_ts)+10


fig, ax = plt.subplots(3, 1, figsize=(14,20))
# Strain
pclip=99
clipVal = np.percentile(np.absolute(strain_rate), pclip)
gca = ax[0]
clb = gca.imshow(strain_rate.T, 
            extent=[das_time[0], das_time[-1], mag_estimate_P.shape[1], 0],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'), interpolation='none')
gca.plot(cvm_tt_tp.flatten(), np.arange(cvm_tt_tp.shape[1]), '-k', linewidth=4)
gca.plot(cvm_tt_ts.flatten(), np.arange(cvm_tt_ts.shape[1]), '-k', linewidth=4)
gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=strain_rate.shape[1], ymin=0, label='earliest tP and tS', color='b')
gca.vlines(x=[np.max(cvm_tt_tp)+2, np.max(cvm_tt_ts)+2], ymax=strain_rate.shape[1], ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s')

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
# axins1.xaxis.set_ticks_position("bottom")


#fig.colorbar(clb, ax=gca, label='strain rate ($10^{-6}$/s)')

# Magnitude
# cmap = plt.cm.Spectral_r  # define the colormap
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]
# # force the first color entry to be grey
# # create the new map
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)

cmap = plt.cm.get_cmap('OrRd', 8)

gca = ax[1]
clb = gca.imshow(mag_estimate_final.T, 
            extent=[das_time[0], das_time[-1], mag_estimate_final.shape[1], 0],
            aspect='auto', vmin=0, vmax=8, cmap=cmap, interpolation='none')
gca.plot(cvm_tt_tp.flatten(), np.arange(cvm_tt_tp.shape[1]), '-k', linewidth=4)
gca.plot(cvm_tt_ts.flatten(), np.arange(cvm_tt_ts.shape[1]), '-k', linewidth=4)
gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=mag_estimate_final.shape[1], ymin=0, label='earliest tP and tS', color='b', zorder=10)
gca.vlines(x=[np.max(cvm_tt_tp)+2, np.max(cvm_tt_ts)+2], ymax=mag_estimate_final.shape[1], ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s', zorder=10)

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.invert_yaxis()

axins2 = inset_axes(gca,
                    width="2%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='lower left')
fig.colorbar(clb, cax=axins2, orientation="vertical", label='Magnitude')
#fig.colorbar(clb, ax=gca, label='Mag.')


gca=ax[2]
# gca.plot(das_time, mag_estimate_final, '-k', linewidth=0.1, alpha=0.1)
gca.plot(das_time, mean_mag, '-r', linewidth=4, alpha=1, zorder=3, label='mean')
gca.plot(das_time, mean_mag-std_mag, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
gca.plot(das_time, mean_mag+std_mag, '--r', linewidth=2, alpha=0.5, zorder=3)
gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=10, ymin=0, label='earliest tP and tS', color='b')
gca.vlines(x=[np.max(cvm_tt_tp)+2, np.max(cvm_tt_ts)+2], ymax=10, ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s')

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

plt.savefig(fig_output_dir + f'/{region_label}_{test_event_id}_estimated_mag_image.png', bbox_inches='tight')
plt.savefig(fig_output_dir + f'/{region_label}_{test_event_id}_estimated_mag_image.pdf', bbox_inches='tight')
# fig.tight_layout()

# %%




# mag_estimate_P_align = (np.log10(data_peak_mat_aligned+1e-12) - site_terms_P - np.log10(distance_to_source)*regP.params['np.log10(distance_in_km)'])/regP.params['magnitude']
# median_mag_P_align = np.nanmedian(mag_estimate_P_align, axis=1)
# std_mag_P_align = np.nanstd(mag_estimate_P_align, axis=1)
# mean_mag_P_align = np.nanmean(mag_estimate_P_align, axis=1)

# mag_estimate_S_align = (np.log10(data_peak_mat_aligned+1e-12) - site_terms_S - np.log10(distance_to_source)*regS.params['np.log10(distance_in_km)'])/regS.params['magnitude']
# std_mag_S_align = np.nanstd(mag_estimate_S_align, axis=1)
# mean_mag_S_align = np.nanmean(mag_estimate_S_align, axis=1)
#%%
# fig, ax = plt.subplots(2,1, figsize=(10, 10))

# gca = ax[0]
# gca.plot(das_time-30, mag_estimate_P_align, '-k', linewidth=0.1, alpha=0.1)
# gca.plot(das_time-30, mean_mag_P_align, '-r', linewidth=2, alpha=0.5, zorder=3, label='mean')
# gca.plot(das_time-30, mean_mag_P_align-std_mag_P_align, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
# gca.plot(das_time-30, mean_mag_P_align+std_mag_P_align, '--r', linewidth=2, alpha=0.5, zorder=3)
# gca.vlines(x=[P_arrival_approx, P_arrival_approx+2], ymax=10, ymin=0, label='tP and tP+2')
# gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')
# gca.set_yticks(np.arange(0, 9))
# gca.set_ylim(0, eq_mag.values*1.4)
# gca.set_xlim(-10, 20)
# #gca.set_xlim(30, 60)
# gca.set_xlabel('Time (s)')
# gca.set_ylabel('Estimated Magnitude')
# gca.set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}, P wave')
# gca.legend(loc=4)

# gca=ax[1]
# gca.plot(das_time-30, mag_estimate_S_align, '-k', linewidth=0.1, alpha=0.1)
# gca.plot(das_time-30, mean_mag_S_align, '-r', linewidth=2, alpha=0.5, zorder=3, label='mean')
# gca.plot(das_time-30, mean_mag_S_align-std_mag_S_align, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
# gca.plot(das_time-30, mean_mag_S_align+std_mag_S_align, '--r', linewidth=2, alpha=0.5, zorder=3)
# gca.vlines(x=[S_arrival_approx, S_arrival_approx+2], ymax=10, ymin=0, label='tS and tS+2')
# gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')
# gca.set_yticks(np.arange(0, 9))
# gca.set_ylim(0, eq_mag.values*1.4)
# gca.set_xlim(-10, 20)
# #gca.set_xlim(30, 60)
# gca.set_xlabel('Time (s)')
# gca.set_ylabel('Estimated Magnitude')
# gca.set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}, S wave')
# gca.legend(loc=4)

# fig.tight_layout()

# plt.savefig(fig_output_dir + f'/{test_event_id}_estimated_mag.png')


# print(f'{test_event_id} done!')
    # except:
    #     continue

#%% unaligned
fig, ax = plt.subplots(3,1, figsize=(10, 15))

gca = ax[0]
gca.plot(das_time, mag_estimate_P, '-k', linewidth=0.1, alpha=0.1)
gca.plot(das_time, mean_mag_P, '-r', linewidth=4, alpha=1, zorder=3, label='mean')
gca.plot(das_time, mean_mag_P-std_mag_P, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
gca.plot(das_time, mean_mag_P+std_mag_P, '--r', linewidth=2, alpha=0.5, zorder=3)
gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=10, ymin=0, label='tP and tS')
gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')
gca.set_yticks(np.arange(0, 9))
gca.set_ylim(0, eq_mag.values*1.4)
# gca.set_xlim(-10, 20)
gca.set_xlim(30, 50)
gca.set_xlabel('Time (s)')
gca.set_ylabel('Estimated Magnitude')
gca.set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}, P wave')
gca.legend(loc=4)

gca=ax[1]
gca.plot(das_time, mag_estimate_S, '-k', linewidth=0.1, alpha=0.1)
gca.plot(das_time, mean_mag_S, '-r', linewidth=4, alpha=1, zorder=3, label='mean')
gca.plot(das_time, mean_mag_S-std_mag_S, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
gca.plot(das_time, mean_mag_S+std_mag_S, '--r', linewidth=2, alpha=0.5, zorder=3)
gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=10, ymin=0, label='tP and tS')
gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')
gca.set_yticks(np.arange(0, 9))
gca.set_ylim(0, eq_mag.values*1.4)
#gca.set_xlim(40, 80)
gca.set_xlim(30, 50)
gca.set_xlabel('Time (s)')
gca.set_ylabel('Estimated Magnitude')
gca.set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}, S wave')
gca.legend(loc=4)

gca=ax[2]
gca.plot(das_time, mag_estimate_final, '-k', linewidth=0.1, alpha=0.1)
gca.plot(das_time, mean_mag, '-r', linewidth=4, alpha=1, zorder=3, label='mean')
gca.plot(das_time, mean_mag-std_mag, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
gca.plot(das_time, mean_mag+std_mag, '--r', linewidth=2, alpha=0.5, zorder=3)

gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=10, ymin=0, label='earlist tP and tS', color='b')
gca.vlines(x=[np.max(cvm_tt_tp)+2, np.max(cvm_tt_ts)+2], ymax=10, ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s')

gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')
gca.set_yticks(np.arange(0, 9))
gca.set_ylim(0, eq_mag.values*1.4)
#gca.set_xlim(40, 80)
gca.set_xlim(30, 50)
gca.set_xlabel('Time (s)')
gca.set_ylabel('Estimated Magnitude')
gca.set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}, P and S')
gca.legend(loc=4)

fig.tight_layout()

plt.savefig(fig_output_dir + f'/{test_event_id}_estimated_mag_unaligned_{region_label}.png')


#%%
fig, ax = plt.subplots(2, 1, figsize=(16,16))
# Strain
pclip=99
clipVal = np.percentile(np.absolute(strain_rate), pclip)
gca = ax[0]
clb = gca.imshow(strain_rate, 
            extent=[0, mag_estimate_P.shape[1], das_time[-1], das_time[0]],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))
gca.plot(np.arange(cvm_tt_tp.shape[1]), cvm_tt_tp.flatten(), '-k', linewidth=4)
gca.plot(np.arange(cvm_tt_ts.shape[1]), cvm_tt_ts.flatten(), '-k', linewidth=4)
gca.set_ylabel('Time (s)')
gca.set_xlabel('channel number')
gca.set_ylim(30, 50)
gca.invert_yaxis()
fig.colorbar(clb, ax=gca, label='strain rate ($10^{-6}$/s)')

# Magnitude
# cmap = plt.cm.Spectral_r  # define the colormap
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]
# # force the first color entry to be grey
# # create the new map
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)

cmap = plt.cm.get_cmap('OrRd', 8)

gca = ax[1]
clb = gca.imshow(mag_estimate_final, 
            extent=[0, mag_estimate_final.shape[1], das_time[-1], das_time[0]],
            aspect='auto', vmin=0, vmax=8, cmap=cmap)
gca.plot(np.arange(cvm_tt_tp.shape[1]), cvm_tt_tp.flatten(), '-k', linewidth=4)
gca.plot(np.arange(cvm_tt_ts.shape[1]), cvm_tt_ts.flatten(), '-k', linewidth=4)
gca.set_ylabel('Time (s)')
gca.set_xlabel('channel number')
gca.set_ylim(30, 50)
gca.invert_yaxis()
fig.colorbar(clb, ax=gca, label='Mag.')
plt.savefig(fig_output_dir + f'/{test_event_id}_estimated_mag_image_{region_label}.png')





#%%
# TEST
def plot_magnitude_estimate_time(mag_estimate, gca):
    cmap = plt.cm.get_cmap('OrRd', 8)
    cmap.set_bad(color='white')
    clb = gca.imshow(mag_estimate.T, 
                extent=[das_time[0], das_time[-1], mag_estimate.shape[1], 0],
                aspect='auto', vmin=-2, vmax=3, cmap=cmap, interpolation='none')
    gca.plot(cvm_tt_tp.flatten(), np.arange(cvm_tt_tp.shape[1]), '-k', linewidth=4)
    gca.plot(cvm_tt_ts.flatten(), np.arange(cvm_tt_ts.shape[1]), '-k', linewidth=4)
    gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=mag_estimate.shape[1], ymin=0, label='earliest tP and tS', color='b', zorder=10)
    gca.vlines(x=[np.max(cvm_tt_tp)+2, np.max(cvm_tt_ts)+2], ymax=mag_estimate.shape[1], ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s', zorder=10)

    gca.set_xlabel('Time (s)')
    gca.set_ylabel('channel number')
    gca.set_xlim(time_rang_show[0], time_rang_show[1])
    gca.invert_yaxis()

    axins2 = inset_axes(gca,
                        width="2%",  # width = 50% of parent_bbox width
                        height="70%",  # height : 5%
                        loc='lower left')
    fig.colorbar(clb, cax=axins2, orientation="vertical", label='Magnitude')
    return gca


time_rang_show = [np.amin(cvm_tt_tp)-3, np.amax(cvm_tt_ts)+5]
fig, ax = plt.subplots(3, 1, figsize=(14,20))

temp = mag_estimate_S*5 + np.arange(0,3926)[np.newaxis, :]

gca = ax[0]
plot_magnitude_estimate_time(mag_estimate_P, gca)
gca = ax[1]
plot_magnitude_estimate_time(mag_estimate_S, gca)
gca = ax[2]
plot_magnitude_estimate_time(mag_estimate_final, gca)