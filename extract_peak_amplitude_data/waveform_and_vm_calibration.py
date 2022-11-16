#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser

import statsmodels.api as sm
import seaborn as sns
import tqdm
import sys

# Work out a handy travel time table to do interpolation
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

sys.path.insert(0, '../')
from utility.general import *
from utility.loading import load_event_data
from utility.processing import remove_outliers, filter_event
from utility.plotting import plot_das_waveforms

import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

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
matplotlib.rcParams.update(params)

import pygmt

#%%
event_folder = '/kuafu/EventData/Arcata_Spring2022'

# load catalog
catalog = pd.read_csv(event_folder + '/catalog.csv')
eq_num = len(catalog)
eq_lat = catalog.latitude # lat
eq_lon = catalog.longitude # lon
eq_mag = catalog.magnitude # catalog magnitude
eq_time = catalog.event_time # eq time

# load the DAS array information
DAS_info = pd.read_csv(event_folder + '/das_info.csv')

DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info.index
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

#%%
tt_output_dir = event_folder + '/theoretical_arrival_time'
mkdir(tt_output_dir)
travel_time_table_file = tt_output_dir + '/travel_time_table.npz'

# from one event to all channels
event_arrival_P = np.zeros((DAS_channel_num, eq_num)) 
event_arrival_S = np.zeros((DAS_channel_num, eq_num)) 

# First look for the precalculated TTT, if not exists, get one from interpolating TauP 
if not os.path.exists(travel_time_table_file):
    model = TauPyModel(model='iasp91')

    # distance list
    distance_fit = np.linspace(0, 1, 100)
    # depth list
    depth_fit = np.arange(0, 100, 1)

    distance_grid, depth_grid = np.meshgrid(distance_fit, depth_fit)


    tavel_time_P_grid = np.zeros(distance_grid.shape)
    tavel_time_S_grid = np.zeros(distance_grid.shape)

    #for i_eq in tqdm.tqdm(range(10), desc="Calculating arrival time..."):
    for i_depth in tqdm.tqdm(range(depth_grid.shape[0]), desc="Calculating arrival time..."):   

        for i_distance in range(distance_grid.shape[1]):
            try:
                arrivals = model.get_ray_paths(depth_fit[i_depth], distance_fit[i_distance], phase_list=['p', 's'])
                tavel_time_P_grid[i_depth, i_distance] = arrivals[0].time
                tavel_time_S_grid[i_depth, i_distance] = arrivals[1].time 
            except:
                tavel_time_P_grid[i_depth, i_distance] = np.nan
                tavel_time_S_grid[i_depth, i_distance] = np.nan

    # save the calculated Travel time table
    np.savez(travel_time_table_file, distance_grid=distance_grid, depth_grid=depth_grid, 
             tavel_time_p_grid=tavel_time_P_grid, tavel_time_s_grid=tavel_time_S_grid)

    print('Travel time table calculated!')
    
    
# The TTT calculated or already exists, directly load it.
temp = np.load(travel_time_table_file)
distance_grid = temp['distance_grid']
depth_grid = temp['depth_grid']
tavel_time_p_grid = temp['tavel_time_p_grid']
tavel_time_s_grid = temp['tavel_time_s_grid']

#%%
# build the interpolation function
from scipy.interpolate import interp2d, griddata
#grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

ii = ~np.isnan(tavel_time_p_grid) # ignore the nan
# interp_f_p = interp2d(distance_grid[ii], depth_grid[ii], tavel_time_p_grid[ii], kind='linear')
# interp_f_s = interp2d(distance_grid[ii], depth_grid[ii], tavel_time_s_grid[ii], kind='linear')

for i_eq in tqdm(range(eq_num), desc="Calculating arrival time..."):   
    # estimate the arrival time of each earthquake to all channels
    P_arrival = np.zeros(DAS_channel_num)
    S_arrival = np.zeros(DAS_channel_num)
    distance_to_source = locations2degrees(DAS_lat, DAS_lon, catalog.iloc[i_eq, :].latitude, catalog.iloc[i_eq, :].longitude)

    P_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_p_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*5))
    S_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_s_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*5))

    # 30s is the event time 
    P_arrival = 30 + P_arrival
    S_arrival = 30 + S_arrival

    DAS_info['P_arrival'] = P_arrival
    DAS_info['S_arrival'] = S_arrival

    DAS_info.to_csv(tt_output_dir + f'/1D_tt_{catalog.iloc[i_eq, :].event_id}.csv', index=False)


#%%
#load event waveform to plot
event_folder = '/kuafu/EventData/Arcata_Spring2022' 
tt_dir = event_folder +  '/theoretical_arrival_time_calibrated' 
catalog = pd.read_csv(event_folder + '/catalog.csv')
das_waveform_path = event_folder + '/data'
DAS_info = pd.read_csv(event_folder + '/das_info.csv')

DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['index']
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

#%%
# work out list to plot
test_event_id_list = []
tp_shift_list, ts_shift_list = [], []

def append_list(test_event_id, tp_shift, ts_shift):
    test_event_id_list.append(test_event_id)
    tp_shift_list.append(tp_shift)
    ts_shift_list.append(ts_shift)
    return test_event_id_list, tp_shift_list, ts_shift_list


test_event_id_list = np.array(catalog.event_id)
tp_shift_list = np.zeros(len(test_event_id_list))
ts_shift_list = np.zeros(len(test_event_id_list))
#%% 
# apply manually TT calibration...
tt_dir0 = event_folder +  '/theoretical_arrival_time' 
tt_dir = event_folder +  '/theoretical_arrival_time_calibrated' 

tp_shift_list = np.zeros(len(test_event_id_list))
tp_shift_list[0:2]=2
tp_shift_list[[2, 7, 11, 12, 20, 22, 23, 24, 25, 44]] =1 
tp_shift_list[8]=0.5
tp_shift_list[45]=0.5

ts_shift_list = np.zeros(len(test_event_id_list))
ts_shift_list[[0, 1, 7, 11, 22]]=3
ts_shift_list[[2, 23, 24, 25, 40, 44]]=2
ts_shift_list[[28, 32, 39]]=1
ts_shift_list[12] = 1.5
ts_shift_list[20] = 2.5
ts_shift_list[45]=0.5

# Apply the calibrated travel time to the model
for i_event in [46,47]:#range(len(test_event_id_list)):
    test_event_id = test_event_id_list[i_event]
    tp_shift = tp_shift_list[i_event]
    ts_shift = ts_shift_list[i_event]

    # plot some waveforms and picking
    VM_picking = pd.read_csv(tt_dir0 + f'/1D_tt_{test_event_id}.csv')
    VM_picking.P_arrival = VM_picking.P_arrival + tp_shift
    VM_picking.S_arrival = VM_picking.S_arrival + ts_shift

    VM_picking.to_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')


#%%
# Output all figures
ymin = 0
ymax = 60

tp_shift_list = np.zeros(len(test_event_id_list))
ts_shift_list = np.zeros(len(test_event_id_list))
# plot waveform
# load the travel time and process
def plot_waveform_with_1D_TT(event_folder, catalog, DAS_index, tt_dir, das_waveform_path, test_event_id, tp_shift, ts_shift, ymin, ymax):
    matplotlib.rcParams.update(params)
    event_info = catalog[catalog.event_id == test_event_id]
    strain_rate, info = load_event_data(das_waveform_path, test_event_id)
    strain_rate = strain_rate[:, DAS_index]
    das_dt = info['dt_s']
    nt = strain_rate.shape[0]
    das_time = np.arange(nt) * das_dt - 30

    # plot some waveforms and picking
    VM_picking = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
    tt_tp = VM_picking.P_arrival - 30 + tp_shift
    tt_ts = VM_picking.S_arrival - 30 + ts_shift


    fig, gca = plt.subplots(figsize=(10, 6))
    plot_das_waveforms(strain_rate, das_time, gca, channel_index=np.array(DAS_index), title=f'{test_event_id}, M{event_info.iloc[0, :].magnitude}', pclip=95, ymin=ymin, ymax=ymax)
    gca.plot(VM_picking['index'], tt_tp, '--k', linewidth=2)
    gca.plot(VM_picking['index'], tt_ts, '-k', linewidth=2)
    #gca.invert_yaxis()

    mkdir(event_folder + '/event_examples_with_1D_TT')
    plt.savefig(event_folder + f'/event_examples_with_1D_TT/{test_event_id}.png', bbox_inches='tight')

i_event = 46
plot_waveform_with_1D_TT(event_folder, catalog, DAS_index, tt_dir, das_waveform_path, test_event_id_list[i_event], tp_shift_list[i_event], ts_shift_list[i_event], ymin, ymax)

# n_eq = len(test_event_id_list)
# with tqdm_joblib(tqdm(desc="File desampling", total=n_eq)) as progress_bar:
#     Parallel(n_jobs=Ncores)(delayed(plot_waveform_with_1D_TT)(event_folder, catalog, DAS_index, tt_dir, das_waveform_path, test_event_id_list[i_event], tp_shift_list[i_event], ts_shift_list[i_event], ymin, ymax) for i_event in range(n_eq))








# # raise


# #%%
# # =========================  Plot both arrays in California with PyGMT ==============================
# gmt_region = [-118.5, -118, 33.75, 34.2]

# projection = "M10c"
# grid = pygmt.datasets.load_earth_relief(resolution="01s", region=gmt_region)

# # calculate the reflection of a light source projecting from west to east
# dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

# fig = pygmt.Figure()
# # define figure configuration
# pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=15, FONT_TITLE="14p,Helvetica,black")

# # --------------- plotting the original Data Elevation Model -----------
# fig.basemap(region=gmt_region, 
# projection=projection, 
# frame=['WSrt+t"LAX"', "xa0.25", "ya0.2"]
# )
# pygmt.makecpt(cmap="geo", series=[-3000, 3000])
# fig.grdimage(
#     grid=grid,
#     projection=projection,
#     cmap=True,
#     shading='+a45+nt1',
#     transparency=40
# )

# fig.plot(x=catalog_fit.longitude.astype('float'), y=catalog_fit.latitude.astype('float'), style="c0.1c", color="red")
# fig.plot(x=catalog_predict.longitude.astype('float'), y=catalog_predict.latitude.astype('float'), style="c0.1c", color="black")
# fig.plot(x=DAS_info_df.longitude[::10].astype('float'), y=DAS_info_df.latitude[::10].astype('float'), pen="1p,blue")

# fig.text(text="Long Valley", x=-119.5, y=38)
# fig.text(text="Ridgecrest", x=-117.5, y=35.4)

# fig.text(text="(b)", x=-120.5, y=39.3)
# fig.show()

# # %%

# %%
