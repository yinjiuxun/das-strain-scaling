#%%
# import numpy as np
import pandas as pd
import numpy as np
import os
import tqdm
import sys

# Work out a handy travel time table to do interpolation
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

sys.path.insert(0, '../')
from utility.general import mkdir

#%% 
event_folder = '/kuafu/EventData/AlumRock5.1/MammothNorth'
# load the DAS array information
DAS_info = pd.read_csv('/kuafu/EventData/Mammoth_north/das_info.csv')


catalog = pd.read_csv(event_folder + '/catalog.csv')
eq_num = len(catalog)
eq_lat = catalog.latitude # lat
eq_lon = catalog.longitude # lon
eq_mag = catalog.magnitude # catalog magnitude
eq_time = catalog.event_time # eq time


DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info.index
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

# calculate tt with taup
tt_output_dir = event_folder + '/theoretical_arrival_time'
mkdir(tt_output_dir)
travel_time_table_file = '/kuafu/EventData/AlumRock5.1/travel_time_table.npz'

# from one event to all channels
event_arrival_P = np.zeros((DAS_channel_num, eq_num)) 
event_arrival_S = np.zeros((DAS_channel_num, eq_num)) 

# First look for the precalculated TTT, if not exists, get one from interpolating TauP 
if not os.path.exists(travel_time_table_file):
    model = TauPyModel(model='iasp91')

    # distance list
    distance_fit = np.linspace(0, 3, 100)
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

for i_eq in tqdm.tqdm(range(eq_num), desc="Calculating arrival time..."):   
    # estimate the arrival time of each earthquake to all channels
    P_arrival = np.zeros(DAS_channel_num)
    S_arrival = np.zeros(DAS_channel_num)
    distance_to_source = locations2degrees(DAS_lat, DAS_lon, catalog.iloc[i_eq, :].latitude, catalog.iloc[i_eq, :].longitude)

    P_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_p_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*catalog.iloc[i_eq, :].depth_km))
    S_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_s_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*catalog.iloc[i_eq, :].depth_km))

    # 30s is the event time 
    P_arrival = 30 + P_arrival
    S_arrival = 30 + S_arrival

    DAS_info['P_arrival'] = P_arrival
    DAS_info['S_arrival'] = S_arrival

    DAS_info.to_csv(tt_output_dir + f'/1D_tt_{catalog.iloc[i_eq, :].event_id}.csv', index=False)
# %%



# # %%
# file_dir = '/kuafu/EventData/Ridgecrest/theoretical_arrival_time'
# files = glob.glob(file_dir + '/1D_tt_*.csv')
# # %%
# for tt_file in files:
#     eq_id = tt_file[-12:-4]
#     temp_pd = pd.read_csv(tt_file)
#     temp_pd = temp_pd[['index','P_arrival','S_arrival']]
#     temp_pd = temp_pd.rename(columns={'index':'ichan','P_arrival':'tp','S_arrival':'ts'})
#     temp_pd['tp'] = temp_pd['tp'] - 30
#     temp_pd['ts'] = temp_pd['ts'] - 30
#     temp_pd.to_csv(file_dir + '/' + eq_id + '.table')
# # %%
