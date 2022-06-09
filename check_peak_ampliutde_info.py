#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser
import obspy

# import the plotting functions
from plotting_functions import *
# import the utility functions
from utility_functions import *

import seaborn as sns

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

import pygmt
#%%
# ==============================  event data ========================================

#%% Specify the file names
# #Ridgecrest
# event_folder = '/kuafu/EventData/Ridgecrest'
# results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
# das_pick_file_folder = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events'
# gmt_region = [-118.5, -117, 35, 36.5]

# # Mammoth North
# event_folder = '/kuafu/EventData/Mammoth_north'
# results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
# das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events'
# gmt_region = [-120, -117.5, 36.5, 39]

# Mammoth South
event_folder = '/kuafu/EventData/Mammoth_south'
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events'
gmt_region = [-120, -117.5, 36.5, 39]

snr_threshold = 10
magnitude_threshold = [2, 10]
#%% Common part of code
DAS_info_files = event_folder + '/das_info.csv'
catalog_file =  event_folder + '/catalog.csv'

das_pick_file_name = '/peak_amplitude.csv'
peak_df_file = das_pick_file_folder + '/' + das_pick_file_name
# Load the peak amplitude results
peak_amplitude_df = pd.read_csv(peak_df_file)
peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.snrP >= snr_threshold) | (peak_amplitude_df.snrS >= snr_threshold)]
peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= magnitude_threshold[0]) & (peak_amplitude_df.magnitude <= magnitude_threshold[1])]

#  load the information about the Ridgecrest DAS channel
DAS_info = pd.read_csv(DAS_info_files)
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['index'].astype('int')
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

catalog_Ridgecrest = pd.read_csv(catalog_file)
# Find events in the pick file
event_id_selected_Ridgecrest = np.unique(peak_amplitude_df['event_id'])
catalog_select = catalog_Ridgecrest[catalog_Ridgecrest.event_id.isin(event_id_selected_Ridgecrest)]
num_events_Ridgecrest = catalog_select.shape[0]
event_lon = catalog_select.longitude
event_lat = catalog_select.latitude
event_id = catalog_select.event_id

# Add the event label for plotting
peak_amplitude_df = add_event_label(peak_amplitude_df)

#%%
# plot time variation of events
time_list = [obspy.UTCDateTime(parser.parse(time)) for time in catalog_select.event_time]
time_span = np.array([time-time_list[0] for time in time_list])
time_span_days = time_span/3600/24

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(time_span_days, catalog_select.magnitude, 'o')
ax.set_xlabel(f'Days from the {time_list[0].isoformat()[:10]}')
ax.set_ylabel('magnitude')
plt.savefig(results_output_dir + '/time_variation_selected_earthquakes.png', bbox_inches='tight')

# plot map
fig, ax = plt.subplots(figsize=(7, 6))
cmp = ax.scatter(DAS_lon, DAS_lat, s=10, c='r')
ax.scatter(event_lon, event_lat, s=10**(catalog_select.magnitude/5), c='k')
#fig.colorbar(cmp)
plt.savefig(results_output_dir + '/map_of_earthquakes_not_grouped.png', bbox_inches='tight')

# plot data statistics
peak_amplitude_df_temp = peak_amplitude_df#.iloc[::5, :]
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km)
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P)
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S)
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S
plt.figure(figsize=(14,8))
g = sns.pairplot(peak_amplitude_df_temp[['magnitude','log10(distance)', 'log10(peak_P)','log10(peak_S)']], kind='hist', diag_kind="kde", corner=True)
g.axes[1,0].set_yticks(np.arange(0,10))
g.axes[1,0].set_ylim((0, 2.3))

g.axes[2,0].set_yticks(np.arange(-3,5))
g.axes[2,0].set_ylim((-2,2))
g.axes[2,1].set_ylim((-2,2))

g.axes[3,0].set_yticks(np.arange(-3,5))
g.axes[3,0].set_ylim((-2,2.5))
g.axes[3,1].set_ylim((-2,2.5))
g.axes[3,2].set_ylim((-2,2.5))
g.axes[3,2].set_xticks(np.arange(-3,5))
g.axes[3,2].set_xlim((-2,2.5))
g.axes[3,3].set_xlim((-2,2.5))

g.axes[2,0].set_xticks(np.arange(0,10))
g.axes[2,0].set_xlim(1.5, 6)
g.axes[2,1].set_xticks(np.arange(0,3,1))
g.axes[2,1].set_xlim((0, 2.3))
g.axes[3,1].set_xlim((0, 2.3))


# Adding annotation
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for gca in g.axes.flatten():
    if gca is not None:
        gca.annotate(f'({letter_list[k]})', xy=(0.05, 0.95), xycoords=gca.transAxes)
        k += 1
        
plt.savefig(results_output_dir + '/data_statistics.png')

#%%
# PyGMT to plot the map with tompography
# Define region for GMT
projection = "M12c"
# Load sample grid (3 arc second global relief) in target area
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=gmt_region)

# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=15)

# --------------- plotting the original Data Elevation Model -----------

fig.basemap(region=gmt_region, 
projection=projection, 
frame=['WSrt+t""', "xa0.5", "ya0.5"]
)
pygmt.makecpt(cmap="dem4", series=[-1000, 3000])
fig.grdimage(
    grid=grid,
    projection=projection,
    cmap=True,
    shading='+a45+nt1',
    transparency=40
)

fig.plot(x=event_lon, y=event_lat, style="c0.15c", color="black")

fig.plot(x=DAS_lon[::10], y=DAS_lat[::10], style="c0.05c", color="red")

fig.show()
fig.savefig(results_output_dir + '/map_of_earthquakes_GMT.png')

#TODO: combine all the regions

#%% 
# Combine all regions (Ridgecrest + Mammoth N and S)
# Specify the file names
combined_results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS'
gmt_region = [-120, -117, 35, 39]

event_folder_list = ['/kuafu/EventData/Ridgecrest', '/kuafu/EventData/Mammoth_north', 
                     '/kuafu/EventData/Mammoth_south', '/kuafu/EventData/Sanriku_ERI']

results_output_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate',
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North',
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South',
                           '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate']

das_pick_file_folder_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events',
                             '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events',
                             '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events',
                             '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']


region_label = ['RC', 'LV-N', 'LV-S', 'Sanriku']

snr_threshold = 10
magnitude_threshold = [2, 10]
#%% Combined all data
peak_amplitude_df_all = pd.DataFrame(columns=['event_id', 'magnitude', 'channel_id', 'distance_in_km', 'snrP', 'snrS',
       'peak_P', 'peak_P_1s', 'peak_P_3s', 'peak_P_4s', 'peak_P_time',
       'peak_P_time_1s', 'peak_P_time_3s', 'peak_P_time_4s', 'peak_S',
       'peak_S_4s', 'peak_S_6s', 'peak_S_8s', 'peak_S_10s', 'peak_S_time',
       'peak_S_time_4s', 'peak_S_time_6s', 'peak_S_time_8s',
       'peak_S_time_10s', 'Array'])
DAS_info_all = pd.DataFrame(columns=['index', 'latitude', 'longitude', 'elevation_m'])
catalog_select_all = pd.DataFrame(columns=['event_id', 'event_time', 'longitude', 'latitude','depth_km','magnitude','magnitude_type','source'])
for i_region in range(len(event_folder_list)):
    event_folder = event_folder_list[i_region]
    results_output_dir = results_output_dir_list[i_region]

    DAS_info_files = event_folder + '/das_info.csv'
    catalog_file =  event_folder + '/catalog.csv'

    das_pick_file_name = '/peak_amplitude.csv'
    peak_df_file = das_pick_file_folder_list[i_region] + '/' + das_pick_file_name
    # Load the peak amplitude results
    peak_amplitude_df = pd.read_csv(peak_df_file)
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.snrP >= snr_threshold) | (peak_amplitude_df.snrS >= snr_threshold)]
    peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= magnitude_threshold[0]) & (peak_amplitude_df.magnitude <= magnitude_threshold[1])]
    peak_amplitude_df['Array'] = region_label[i_region]

    # if i_region == 3: # TODO: due to unknown reason, the Sanriku data need to multiply this...
    #     peak_amplitude_df.peak_S = peak_amplitude_df.peak_S * 1e2

    peak_amplitude_df_all = pd.concat((peak_amplitude_df_all, peak_amplitude_df), axis=0, ignore_index=True)

    #  load the information about the DAS channel
    DAS_info = pd.read_csv(DAS_info_files)
    DAS_info_all = pd.concat((DAS_info_all, DAS_info), axis=0, ignore_index=True)


    catalog = pd.read_csv(catalog_file)
    # Find events in the pick file
    event_id_selected = np.unique(peak_amplitude_df['event_id'])
    catalog_select = catalog[catalog.event_id.isin(event_id_selected)]
    catalog_select_all = pd.concat((catalog_select_all, catalog_select), axis=0, ignore_index=True)


DAS_channel_num = DAS_info_all.shape[0]
DAS_index = DAS_info_all['index'].astype('int')
DAS_lon = DAS_info_all.longitude
DAS_lat = DAS_info_all.latitude


num_events = catalog_select_all.shape[0]
event_lon = catalog_select_all.longitude
event_lat = catalog_select_all.latitude
event_id = catalog_select_all.event_id

# TODO: load the results from combined folder...

#%%
# plot time variation of events
time_list = [obspy.UTCDateTime(parser.parse(time)) for time in catalog_select_all.event_time]
time_span = np.array([time-time_list[0] for time in time_list])
time_span_days = time_span/3600/24

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(time_span_days, catalog_select_all.magnitude, 'o')
ax.set_xlabel(f'Days from the {time_list[0].isoformat()[:10]}')
ax.set_ylabel('magnitude')
# TODO: add two horizontal line indicating the deployment time of arrays
plt.savefig(combined_results_output_dir + '/time_variation_selected_earthquakes.png', bbox_inches='tight')

# plot time variation of events in vertical
fig, ax = plt.subplots(figsize=(6, 12))
ax.plot(catalog_select_all.magnitude, time_span_days, 'o')
ax.vlines(x=8, ymin=0, ymax=696, color='r', label='Ridgecrest', linewidth=5)
ax.vlines(x=7.5, ymin=467, ymax=time_span_days[-1], color='green', label='Long Valley', linewidth=5)

ax.set_ylabel(f'Days from the {time_list[0].isoformat()[:10]}')
ax.set_xlabel('magnitude')
ax.legend(loc=4)
# TODO: add two horizontal line indicating the deployment time of arrays
plt.savefig(combined_results_output_dir + '/time_variation_selected_earthquakes_vertical.png', bbox_inches='tight')


#%% plot data statistics
add_possible_clipping = True
peak_amplitude_df_temp = peak_amplitude_df_all.iloc[::1, :]
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P.astype('float'))
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float'))
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S


fig = plt.figure(figsize=(14,8))

g = sns.PairGrid(peak_amplitude_df_temp[['magnitude','log10(distance)', 'log10(peak_P)','log10(peak_S)', 'Array']], 
                 diag_sharey=False, corner=True, hue='Array', palette='Set2', height=4, aspect=0.8)


g.map_diag(sns.kdeplot, hue=None, color=".1", linewidth=2) #hue='Array', palette='Set2', 
g.map_diag(sns.kdeplot, linewidth=2, alpha=1) #hue='Array', palette='Set2', 

g.map_lower(sns.histplot, pmax=0.5, hue=None, color=".1", cbar=True, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0})
g.map_lower(sns.histplot, pmax=0.5, alpha=0.3)

plt.subplots_adjust(hspace=0.2, wspace=0.25)

g.add_legend()
g._legend.set_bbox_to_anchor((0.3, 0.9))


# This part is a little tricky for now.
if add_possible_clipping:
    M_clipping = np.arange(2, 9)
    # P clipping
    D_clipping_RC_P = M_clipping*5/14 - (2-0.615)/1.4
    D_clipping_LV_N_P = M_clipping*5/14 - (1.5-0.391)/1.4
    D_clipping_LV_S_P = M_clipping*5/14 - (1.5-0.277)/1.4

    # # S clipping
    # D_clipping_RC_S = M_clipping*0.57/1.21 - (2-0.615)/1.21
    # D_clipping_LV_N_S = M_clipping*0.57/1.21 - (2-0.391)/1.21
    # D_clipping_LV_S_S = M_clipping*0.57/1.21 - (2-0.277)/1.21

    g.axes[1,0].plot(M_clipping, D_clipping_RC_P, '-', color='#7DC0A6', linewidth=2.5)
    g.axes[1,0].plot(M_clipping, D_clipping_LV_N_P, '-', color='#ED926B', linewidth=2.5)
    g.axes[1,0].plot(M_clipping, D_clipping_LV_S_P, '-', color='#91A0C7', linewidth=2.5)

# show the Sanriku data
g.axes[1,0].plot(peak_amplitude_df.magnitude, np.log10(peak_amplitude_df.distance_in_km), '.', markersize=1, color='#DA8EC0', alpha=0.5)
g.axes[3,0].plot(peak_amplitude_df.magnitude, np.log10(peak_amplitude_df.peak_S), '.', markersize=1, color='#DA8EC0', alpha=0.5)
g.axes[3,1].plot(np.log10(peak_amplitude_df.distance_in_km), np.log10(peak_amplitude_df.peak_S), '.', markersize=1, color='#DA8EC0', alpha=0.5)

g.axes[0,0].set_title('magnitude')
g.axes[0,0].tick_params(labelbottom=True)
g.axes[1,1].set_title('log10(distance)')
g.axes[1,1].tick_params(labelbottom=True)
g.axes[2,2].set_title('log10(peak_P)')
g.axes[2,2].tick_params(labelbottom=True)
g.axes[3,3].set_title('log10(peak_S)')
g.axes[3,3].tick_params(labelbottom=True)

g.axes[1,0].set_yticks(np.arange(0,10))
g.axes[1,0].set_ylim((0, 3))

g.axes[2,0].set_yticks(np.arange(-3,5))
g.axes[2,0].set_ylim((-2,2.5))
g.axes[2,1].set_ylim((-2,2.5))

g.axes[3,0].set_yticks(np.arange(-3,5))
g.axes[3,0].set_ylim((-2,2.5))
g.axes[3,1].set_ylim((-2,2.5))
g.axes[3,2].set_ylim((-2,2.5))
g.axes[3,2].set_xticks(np.arange(-3,5))
g.axes[3,2].set_xlim((-2,2.5))
g.axes[3,3].set_xlim((-2,2.5))

g.axes[2,0].set_xticks(np.arange(0,10))
g.axes[2,0].set_xlim(1.5, 6)
g.axes[2,1].set_xticks(np.arange(0,3,1))
g.axes[2,1].set_xlim((0, 3))
g.axes[3,1].set_xlim((0, 3)) # 2.3

# Adding annotation
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for gca in g.axes.flatten():
    if gca is not None:
        gca.annotate(f'({letter_list[k]})', xy=(-0.2, 1.0), xycoords=gca.transAxes)
        k += 1

plt.tight_layout()

if add_possible_clipping:       
    plt.savefig(combined_results_output_dir + '/data_correlation_clipping.png', bbox_inches='tight')
else:
    plt.savefig(combined_results_output_dir + '/data_correlation.png', bbox_inches='tight')


#%%
# Plot both arrays
projection = "M12c"
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=gmt_region)

# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=15)

# --------------- plotting the original Data Elevation Model -----------

fig.basemap(region=gmt_region, 
projection=projection, 
frame=['WSrt+t""', "xa0.5", "ya0.5"]
)
pygmt.makecpt(cmap="dem4", series=[-1000, 3000])
fig.grdimage(
    grid=grid,
    projection=projection,
    cmap=True,
    shading='+a45+nt1',
    transparency=40
)

fig.plot(x=catalog_select_all.longitude.astype('float'), y=catalog_select_all.latitude.astype('float'), style="c0.1c", color="black")

fig.plot(x=DAS_info_all.longitude[::10].astype('float'), y=DAS_info_all.latitude[::10].astype('float'), style="c0.1c", color="blue")

fig.text(text="Mammoth", x=-119.5, y=38)
fig.text(text="Ridgecrest", x=-117.5, y=35.4)

# Create an inset map, setting the position to bottom right, the width to
# 3 cm, the height to 3.6 cm, and the x- and y-offsets to
# 0.1 cm, respectively. Draws a rectangular box around the inset with a fill
# color of "white" and a pen of "1p".
with fig.inset(position="jBL+w4.3c/7.5c+o0.1c", box="+gwhite+p1p"):
    # Plot the Japan main land in the inset using coast. "U54S/?" means UTM
    # projection with map width automatically determined from the inset width.
    # Highlight the Japan area in "lightbrown"
    # and draw its outline with a pen of "0.2p".
    fig.coast(
        region=[-125, -115, 30, 45],
        projection="J",
        #land='gray',#
        dcw="US+ggray+p0.2p",
        #water='lightblue',
        area_thresh=10000,
        resolution='f',
        borders=["1/0.5p,black", "2/0.25p,black"]
    )
    # Plot a rectangle ("r") in the inset map to show the area of the main
    # figure. "+s" means that the first two columns are the longitude and
    # latitude of the bottom left corner of the rectangle, and the last two
    # columns the longitude and latitude of the uppper right corner.
    rectangle = [[gmt_region[0], gmt_region[2], gmt_region[1], gmt_region[3]]]
    fig.plot(data=rectangle, style="r+s", pen="2p,red")

fig.show()
fig.savefig(combined_results_output_dir + '/map_of_earthquakes_GMT.png')













#%% Temporal use for Japan data

# Japan submarine
event_folder = '/kuafu/EventData/Sanriku_ERI'
results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events'
gmt_region = [140, 145, 35, 42]

snr_threshold = 10
magnitude_threshold = [2, 10]

#%% Common part of code
DAS_info_files = event_folder + '/das_info.csv'
catalog_file =  event_folder + '/catalog.csv'

das_pick_file_name = '/peak_amplitude.csv'
peak_df_file = das_pick_file_folder + '/' + das_pick_file_name
# Load the peak amplitude results
peak_amplitude_df = pd.read_csv(peak_df_file)
peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.snrP >= snr_threshold) | (peak_amplitude_df.snrS >= snr_threshold)]
peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= magnitude_threshold[0]) & (peak_amplitude_df.magnitude <= magnitude_threshold[1])]

#  load the information about the Ridgecrest DAS channel
DAS_info = pd.read_csv(DAS_info_files)
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['index'].astype('int')
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

catalog = pd.read_csv(catalog_file)
# Find events in the pick file
event_id_selected = np.unique(peak_amplitude_df['event_id'])
catalog_select = catalog[catalog.event_id.isin(event_id_selected)]
num_events = catalog_select.shape[0]
event_lon = catalog_select.longitude
event_lat = catalog_select.latitude
event_id = catalog_select.event_id

# Add the event label for plotting
peak_amplitude_df = add_event_label(peak_amplitude_df)

#%%
# plot time variation of events
time_list = [obspy.UTCDateTime(parser.parse(time)) for time in catalog_select.event_time]
time_span = np.array([time-time_list[0] for time in time_list])
time_span_days = time_span/3600/24

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(time_span_days, catalog_select.magnitude, 'o')
ax.set_xlabel(f'Days from the {time_list[0].isoformat()[:10]}')
ax.set_ylabel('magnitude')
plt.savefig(results_output_dir + '/time_variation_selected_earthquakes.png', bbox_inches='tight')

# plot map
fig, ax = plt.subplots(figsize=(7, 6))
cmp = ax.scatter(DAS_lon, DAS_lat, s=10, c='r')
ax.scatter(event_lon, event_lat, s=10**(catalog_select.magnitude/5), c='k')
#fig.colorbar(cmp)
plt.savefig(results_output_dir + '/map_of_earthquakes_not_grouped.png', bbox_inches='tight')

# plot data statistics
peak_amplitude_df_temp = peak_amplitude_df#.iloc[::5, :]
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km)
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P) + 2
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S) + 2
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S
plt.figure(figsize=(14,8))
g = sns.pairplot(peak_amplitude_df_temp[['magnitude','log10(distance)', 'log10(peak_P)','log10(peak_S)']], kind='hist', diag_kind="kde", corner=True)
g.axes[1,0].set_yticks(np.arange(0,10))
g.axes[1,0].set_ylim((1, 3))

g.axes[2,0].set_yticks(np.arange(-3,5))
g.axes[2,0].set_ylim((-2,2))
g.axes[2,1].set_ylim((-2,2))

g.axes[3,0].set_yticks(np.arange(-3,5))
g.axes[3,0].set_ylim((-2,2.5))
g.axes[3,1].set_ylim((-2,2.5))
g.axes[3,2].set_ylim((-2,2.5))
g.axes[3,2].set_xticks(np.arange(-3,5))
g.axes[3,2].set_xlim((-2,2.5))
g.axes[3,3].set_xlim((-2,2.5))

g.axes[2,0].set_xticks(np.arange(0,10))
g.axes[2,0].set_xlim(1.5, 6)
g.axes[2,1].set_xticks(np.arange(0,3,1))
g.axes[2,1].set_xlim((0, 2.3))
g.axes[3,1].set_xlim((1, 3))


# Adding annotation
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for gca in g.axes.flatten():
    if gca is not None:
        gca.annotate(f'({letter_list[k]})', xy=(0.05, 0.95), xycoords=gca.transAxes)
        k += 1
        
plt.savefig(results_output_dir + '/data_statistics.png')

#%%
# Plot both arrays
projection = "M12c"
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=gmt_region)

# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=15)

# --------------- plotting the original Data Elevation Model -----------

fig.basemap(region=gmt_region, 
projection=projection, 
frame=['WSrt+t""', "xa2.0", "ya2.0"]
)
pygmt.makecpt(cmap="geo", series=[-8000, 8000])
fig.grdimage(
    grid=grid,
    projection=projection,
    cmap=True,
    shading='+a45+nt1',
    transparency=20
)

fig.plot(x=catalog_select.longitude.astype('float'), y=catalog_select.latitude.astype('float'), style="c0.2c", color="black")

# show the werid 4130 event
event_4130 = catalog_select[catalog_select.event_id == 4130]
fig.plot(x=event_4130.longitude.astype('float'), y=event_4130.latitude.astype('float'), style="c0.2c", color="red")
fig.text(text="4130", x=event_4130.longitude.astype('float')+0.15, y=event_4130.latitude.astype('float'),color="red")

fig.plot(x=DAS_info.longitude[::10], y=DAS_info.latitude[::10], style="c0.1c", color="blue")

# fig.text(text="Mammoth", x=-119.5, y=38)
# fig.text(text="Ridgecrest", x=-117.5, y=35.4)

# Create an inset map, setting the position to bottom right, the width to
# 3 cm, the height to 3.6 cm, and the x- and y-offsets to
# 0.1 cm, respectively. Draws a rectangular box around the inset with a fill
# color of "white" and a pen of "1p".
with fig.inset(position="jBL+w4.3c/5.0c+o0.1c", box="+gwhite+p1p"):
    # Plot the Japan main land in the inset using coast. "U54S/?" means UTM
    # projection with map width automatically determined from the inset width.
    # Highlight the Japan area in "lightbrown"
    # and draw its outline with a pen of "0.2p".
    fig.coast(
        region=[129, 146, 30, 46],
        projection="U54S/?",
        #land='gray',#
        dcw="JP+ggray+p0.2p",
        #water='lightblue',
        area_thresh=10000,
        resolution='f',
        #borders=["1/0.5p,black", "2/0.25p,black"]
    )
    # Plot a rectangle ("r") in the inset map to show the area of the main
    # figure. "+s" means that the first two columns are the longitude and
    # latitude of the bottom left corner of the rectangle, and the last two
    # columns the longitude and latitude of the uppper right corner.
    rectangle = [[gmt_region[0], gmt_region[2], gmt_region[1], gmt_region[3]]]
    fig.plot(data=rectangle, style="r+s", pen="2p,red")

fig.show()
fig.savefig(results_output_dir + '/map_of_earthquakes_GMT.png')



# %%
