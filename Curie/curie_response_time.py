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

from scipy.interpolate import griddata
from obspy.geodetics import locations2degrees
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

event_folder = '/kuafu/EventData/Curie'#'/kuafu/EventData/Mammoth_south' #'/kuafu/EventData/Ridgecrest'
tt_dir = event_folder +  '/theoretical_arrival_time0' ##
test_event_id =  9001#9001 #
tt_shift_p, tt_shift_s = 0, 0
given_range_P = None
given_range_S = None

catalog = pd.read_csv(event_folder + '/catalog.csv')
das_waveform_path = event_folder + '/data_raw'

#%%
# 2. Specify the array to look at
# load the DAS array information
DAS_info = pd.read_csv(event_folder + '/das_info.csv')
# Only keep the first 7000 channels
DAS_info = DAS_info[DAS_info['index'] < 7000]

# DAS_info = pd.read_csv('/kuafu/EventData/Mammoth_south/das_info.csv')
# specify the directory of ML picking
ML_picking_dir = event_folder + '/picks_phasenet_das'
use_ML_picking = True
if use_ML_picking:
    picking_label='ml'
else:
    picking_label='vm'

#%%
# 3. Specify the coefficients to load
# regression coefficients of the multiple array case
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/'
regression_dir = f'iter_regression_results_smf{weight_text}_100_channel_at_least'

# make figure output directory
fig_output_dir = f'/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate/transfer_regression_specified_smf{weight_text}_100_channel_at_least_9007/estimated_M_7000'
if not os.path.exists(fig_output_dir):
    os.mkdir(fig_output_dir)

#%%
# Load DAS channels
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['index']
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

# Load DAS data
strain_rate, info = load_event_data(das_waveform_path, test_event_id)
strain_rate = strain_rate[:, DAS_index]
das_dt = info['dt_s']
nt = strain_rate.shape[0]
das_time0 = np.arange(nt) * das_dt-30

# Load nearby stations
stations = pd.read_csv('/kuafu/yinjx/Curie/curie_nearby_stations.csv', index_col=None)
STA_lon = stations.Longitude
STA_lat = stations.Latitude

DAS_channel_num = len(DAS_info)
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
eq_depth = eq_info.depth_km # eq depth

# get distance from earthquake to each channel
distance_to_source0 = locations2degrees(DAS_lat, DAS_lon, eq_lat, eq_lon) * 113 # in km
distance_to_source0 = np.sqrt(eq_info.iloc[0, :].depth_km**2 + distance_to_source0**2)
distance_to_source0 = distance_to_source0[np.newaxis, :]

# get distance from earthquake to all the regional stations
distance_to_source_sta = locations2degrees(STA_lat, STA_lon, eq_lat, eq_lon) 

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

        ML_picking_P = ML_picking_P[ML_picking_P.channel_index<=DAS_index.max()]
        ML_picking_S = ML_picking_S[ML_picking_S.channel_index<=DAS_index.max()]

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

    cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
    tt_tp = np.array(cvm_tt.P_arrival)
    tt_tp = tt_tp[np.newaxis, :]
    tt_ts = np.array(cvm_tt.S_arrival)
    tt_ts = tt_ts[np.newaxis, :]

    # For travel time from velocity model, may need some manual correction
    tt_tp= tt_tp+tt_shift_p 
    tt_ts= tt_ts+tt_shift_s

# %%
# The TTT calculated or already exists, directly load it.
travel_time_table_file = '/kuafu/yinjx/Curie/arrival_time_matching/travel_time_table.npz'
temp = np.load(travel_time_table_file)
distance_grid = temp['distance_grid']
depth_grid = temp['depth_grid']
tavel_time_p_grid = temp['tavel_time_p_grid']
tavel_time_s_grid = temp['tavel_time_s_grid']

P_arrival_STA = griddata(np.array([distance_grid.flatten(), depth_grid.flatten()]).T, tavel_time_p_grid.flatten(), (distance_to_source_sta, np.ones(distance_to_source_sta.shape)*eq_depth.iloc[0]))
S_arrival_STA = griddata(np.array([distance_grid.flatten(), depth_grid.flatten()]).T, tavel_time_s_grid.flatten(), (distance_to_source_sta, np.ones(distance_to_source_sta.shape)*eq_depth.iloc[0]))

sorted_P_arrival = np.sort(P_arrival_STA)

n_sta = 4
STA_response_time = sorted_P_arrival[n_sta-1]

# %%
