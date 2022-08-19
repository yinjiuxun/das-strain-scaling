#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

from obspy.geodetics import locations2degrees

# import the plotting functions
from plotting_functions import *
# import the utility functions
from utility_functions import *
import matplotlib as mpl

#%% Define functions
# Use the predicted strain to calculate magnitude
def estimate_magnitude_from_data(peak_amplitude_df, reg, type, fitting_type='without_site', site_term_column='region_site'):
    if fitting_type == 'with_site':
        # get the annoying categorical keys
        try:
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
        except:
            raise NameError(f"Index {site_term_column} doesn't exist!")

        if type == 'P':    
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')

    elif fitting_type == 'with_attenuation':
        # get the annoying categorical keys C(region)[ridgecrest]:distance_in_km 
        try:
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
            site_attenu_keys = np.array([f'C(region)[{region}]:distance_in_km' for region in peak_amplitude_df['region']])
        except:
            raise NameError(f"Index {site_term_column} doesn't exist!")

        if type == 'P':    
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km) \
                        - np.array(reg.params[site_attenu_keys]) * np.array(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km) \
                        - np.array(reg.params[site_attenu_keys]) * np.array(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')
        
    elif fitting_type == 'without_site':
        if type == 'P':
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - reg.params['Intercept'] \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - reg.params['Intercept'] \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')

    else:
        raise NameError('Fitting type is undefined!')
        
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

# ========================== work on Combined results from RM ================================
# First check how well the regression relation can be used to calculate Magnitude
# Then try to use the regression relation from small events to predict the larger ones
#%% load the results from combined regional site terms t
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/'#'/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RO'
regression_dir = 'regression_results_smf_weighted_100_channel_at_least'
site_term_column = 'region_site'
fitting_type = 'with_site'
nearby_channel_number = 10

region_label = 'ridgecrest' #'mammothN' #
event_folder = '/kuafu/EventData/Ridgecrest' #'/kuafu/EventData/Mammoth_north'#'/kuafu/EventData/Mammoth_south'
tt_dir = event_folder +  '/theoretical_arrival_time' ##'/model_proc_tt/CVM3D'
test_event_id = 40063391 #40063391(M4.57) #73584926(M6) 73481241(M5) 39493944(M5.8) 73585021(M4.6)

# load catalog
catalog = pd.read_csv(event_folder + '/catalog.csv')
# load the DAS array information
DAS_info = pd.read_csv(event_folder + '/das_info.csv')

DAS_channel_num = DAS_info.shape[0]
# DAS_index = DAS_info['index'].astype('int')
DAS_index = DAS_info.index
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

das_path = event_folder + '/data'
ml_pick_dir = event_folder + '/picks_phasenet_das'

#%% make figure output directory
fig_output_dir = results_output_dir + '/' + regression_dir + '/estimated_M'
if not os.path.exists(fig_output_dir):
    os.mkdir(fig_output_dir)

#%% # ========================== Real time estimation ================================
# load regression parameters
regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")

#%%
# # Load event information 
# peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
# event_peak_df = peak_amplitude_df[peak_amplitude_df.magnitude > 5.5]
# event_id_list = event_peak_df.event_id.unique().astype('int')

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

# theoretical travel time 
if 'mammoth' in region_label:
    cvm_tt = pd.read_csv(tt_dir + f'/{test_event_id}.csv')
    cvm_tt_tp = np.array(cvm_tt.tp)+30
    cvm_tt_tp = cvm_tt_tp[np.newaxis, :]
    cvm_tt_ts = np.array(cvm_tt.ts)+30
    cvm_tt_ts = cvm_tt_ts[np.newaxis, :]
elif 'ridgecrest' in region_label:
    cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
    cvm_tt_tp = np.array(cvm_tt.P_arrival)
    cvm_tt_tp = cvm_tt_tp[np.newaxis, :]
    cvm_tt_ts = np.array(cvm_tt.S_arrival)
    cvm_tt_ts = cvm_tt_ts[np.newaxis, :]

# get distance from earthquake to each channel
distance_to_source = locations2degrees(DAS_lat, DAS_lon, eq_lat, eq_lon) * 113 # in km
distance_to_source = distance_to_source[np.newaxis, :]

strain_rate_clipped_P = strain_rate.copy()
das_time1 = das_time0[:, np.newaxis]
strain_rate_clipped_P[das_time1<=cvm_tt_tp]=np.nan
strain_rate_clipped_P[(das_time1>cvm_tt_tp+2)] = np.nan

strain_rate_clipped_S = strain_rate.copy()
strain_rate_clipped_S[(das_time1<cvm_tt_ts)] = np.nan

time_step = 10
data_peak_mat_P = np.zeros((np.ceil(strain_rate.shape[0]/time_step).astype('int'), strain_rate.shape[1]))
data_peak_mat_S = np.zeros((np.ceil(strain_rate.shape[0]/time_step).astype('int'), strain_rate.shape[1]))
for i, ii_win in enumerate(range(0, strain_rate.shape[0], time_step)):
    # data_peak_mat[i, :] = np.nanmax(abs(strain_rate_clipped[ii_win:(ii_win+time_step), :]), axis=0)  

    # keep the maximum unchanged
    data_peak_mat_P[i, :] = np.nanmax(abs(strain_rate_clipped_P[0:(ii_win+time_step), :]), axis=0)     
    data_peak_mat_S[i, :] = np.nanmax(abs(strain_rate_clipped_S[0:(ii_win+time_step), :]), axis=0)    

das_time = das_time0[::time_step]


# das_time_channels = das_time[:, np.newaxis] - cvm_tt_tp

# data_peak_mat_aligned = np.zeros(data_peak_mat.shape)
# for i in range(data_peak_mat.shape[1]):
#     data_peak_mat_aligned[:, i] = np.interp(das_time-31.5, das_time_channels[:, i], data_peak_mat[:, i], right=np.nan, left=np.nan)
# ml_picks = pd.read_csv(ml_pick_dir + f'/{test_event_id}.csv')

# # extract the picked information
# ml_picks_p = ml_picks[ml_picks['phase_type'] == 'P']
# ml_picks_s = ml_picks[ml_picks['phase_type'] == 'S']
# P_arrival_index = np.median(ml_picks_p.phase_index).astype('int')
# S_arrival_index = np.median(ml_picks_s.phase_index).astype('int')

# P_arrival_approx = das_time0[P_arrival_index]
# S_arrival_approx = das_time0[S_arrival_index]

# get the site terms
if nearby_channel_number == -1: # when nearby_channel_number == -1, combined all channels!
    nearby_channel_number = DAS_index.max()+1
    temp1= np.arange(0, DAS_index.max())
else:
    temp1= np.arange(0, DAS_index.max()+1) # original channel number
temp2 = DAS_index // nearby_channel_number # combined channel number

site_term_keys = np.array([f'C(region_site)[{region_label}-{site_term}]' for site_term in temp2])

#site_terms = regP.params[site_term_keys]
site_terms_P = np.zeros(site_term_keys.shape)
site_terms_S = np.zeros(site_term_keys.shape)
for ii, site_term_key in enumerate(site_term_keys):
    if site_term_key in regP.params.keys():
        site_terms_P[ii] = regP.params[site_term_key]
    else:
        site_terms_P[ii] = np.nan

    if site_term_key in regS.params.keys():
        site_terms_S[ii] = regS.params[site_term_key]
    else:
        site_terms_S[ii] = np.nan


#site_terms = site_terms[DAS_index]
site_terms_P = site_terms_P[np.newaxis, :]
site_terms_S = site_terms_S[np.newaxis, :]

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


#%% Having all together as one figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, ax = plt.subplots(3, 1, figsize=(14,20))
# Strain
pclip=99
clipVal = np.percentile(np.absolute(strain_rate), pclip)
gca = ax[0]
clb = gca.imshow(strain_rate.T, 
            extent=[das_time[0], das_time[-1], mag_estimate_P.shape[1], 0],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))
gca.plot(cvm_tt_tp.flatten(), np.arange(cvm_tt_tp.shape[1]), '-k', linewidth=4)
gca.plot(cvm_tt_ts.flatten(), np.arange(cvm_tt_ts.shape[1]), '-k', linewidth=4)
gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=1150, ymin=0, label='earliest tP and tS', color='b')
gca.vlines(x=[np.max(cvm_tt_tp)+2, np.max(cvm_tt_ts)+2], ymax=1150, ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s')

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(31, 40)
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
            aspect='auto', vmin=0, vmax=8, cmap=cmap)
gca.plot(cvm_tt_tp.flatten(), np.arange(cvm_tt_tp.shape[1]), '-k', linewidth=4)
gca.plot(cvm_tt_ts.flatten(), np.arange(cvm_tt_ts.shape[1]), '-k', linewidth=4)
gca.vlines(x=[np.min(cvm_tt_tp), np.min(cvm_tt_ts)], ymax=1150, ymin=0, label='earliest tP and tS', color='b', zorder=10)
gca.vlines(x=[np.max(cvm_tt_tp)+2, np.max(cvm_tt_ts)+2], ymax=1150, ymin=0, linestyle='--', color='b', label='latest tP+2s and tS+2s', zorder=10)

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(31, 40)
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
gca.set_xlim(31, 40)
#gca.set_xlim(30, 60)
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

plt.savefig(fig_output_dir + f'/{test_event_id}_estimated_mag_image_{region_label}_all.png', bbox_inches='tight')
plt.savefig(fig_output_dir + f'/{test_event_id}_estimated_mag_image_{region_label}_all.pdf', bbox_inches='tight')
# fig.tight_layout()

# %%
