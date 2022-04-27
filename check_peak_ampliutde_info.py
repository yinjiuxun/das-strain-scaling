#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np

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
# %%
# ==============================  Ridgecrest data ========================================
#% Specify the file names
DAS_info_files = '/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Ridgecrest_ODH3.txt'
catalog_file =  '/home/yinjx/notebooks/strain_scaling/Ridgecrest_das_catalog_M2_M8.txt'
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
das_pick_file_name = '/peak_amplitude_M3+.csv'
peak_df_file = results_output_dir + '/' + das_pick_file_name
# Load the peak amplitude results
peak_amplitude_df_Ridgecrest = pd.read_csv(peak_df_file)

#  load the information about the Ridgecrest DAS channel
DAS_info = np.genfromtxt(DAS_info_files)
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info[:, 0].astype('int')
DAS_lon_Ridgecrest = DAS_info[:, 1]
DAS_lat_Ridgecrest = DAS_info[:, 2]

catalog_Ridgecrest = pd.read_csv(catalog_file, sep='\s+', header=None, skipfooter=1, engine='python')
# Find events in the pick file
event_id_selected_Ridgecrest = np.unique(peak_amplitude_df_Ridgecrest['event_id'])
catalog_select_Ridgecrest = catalog_Ridgecrest[catalog_Ridgecrest[0].isin(event_id_selected_Ridgecrest)]
num_events_Ridgecrest = catalog_select_Ridgecrest.shape[0]
event_lon_Ridgecrest = np.array(catalog_select_Ridgecrest[5])
event_lat_Ridgecrest = np.array(catalog_select_Ridgecrest[4])
event_id = np.array(catalog_select_Ridgecrest[0])

catalog_select_Ridgecrest

# Add the event label for plotting
peak_amplitude_df_Ridgecrest = add_event_label(peak_amplitude_df_Ridgecrest)

#%%
# plot time variation of events
time_list_Ridgecrest = [obspy.UTCDateTime(time) for time in catalog_select_Ridgecrest[3]]
time_span_Ridgecrest = np.array([time-time_list_Ridgecrest[0] for time in time_list])
time_span_days_Ridgecrest = time_span_Ridgecrest/3600/24

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(time_span_days_Ridgecrest, catalog_select_Ridgecrest[7], 'o')
ax.set_xlabel(f'Days from the {time_list_Ridgecrest[0].isoformat()[:10]}')
ax.set_ylabel('magnitude')
plt.savefig(results_output_dir + '/time_variation_selected_earthquakes.png', bbox_inches='tight')

# plot map
fig, ax = plt.subplots(figsize=(7, 6))
cmp = ax.scatter(DAS_lon_Ridgecrest, DAS_lat_Ridgecrest, s=10, c='r')
ax.scatter(event_lon_Ridgecrest, event_lat_Ridgecrest, s=10**(catalog_select_Ridgecrest[7]/5), c='k')
#fig.colorbar(cmp)
plt.savefig(results_output_dir + '/map_of_earthquakes_not_grouped.png', bbox_inches='tight')

# plot data statistics
peak_amplitude_df_temp = peak_amplitude_df_Ridgecrest.iloc[::5, :]
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km)
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P)
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S
plt.figure(figsize=(14,8))
sns.pairplot(peak_amplitude_df_temp[['magnitude','log10(distance)', 'max_P_time', 'log10(peak_P)','P/S']], kind='hist', diag_kind="kde", corner=True)
plt.savefig(results_output_dir + '/ridgecrest_data_statistics.png')

#%%
# Define region of interest around Ridgecrest
region = [-118.5, -117, 35, 36.5]
projection = "M12c"
# Load sample grid (3 arc second global relief) in target area
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=region)

# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=15)

# --------------- plotting the original Data Elevation Model -----------

fig.basemap(region=region, 
projection=projection, 
frame=['WSrt+t"Ridgecrest"', "xa0.5", "ya0.5"]
)
pygmt.makecpt(cmap="dem4", series=[-1000, 3000])
fig.grdimage(
    grid=grid,
    projection=projection,
    cmap=True,
    shading='+a45+nt1',
    transparency=40
)

fig.plot(x=event_lon_Ridgecrest, y=event_lat_Ridgecrest, style="c0.15c", color="black")

fig.plot(x=DAS_lon_Ridgecrest[::10], y=DAS_lat_Ridgecrest[::10], style="c0.05c", color="red")

fig.show()
fig.savefig(results_output_dir + '/map_of_earthquakes_GMT_Ridgecrest.png')

#%%
# Looking into the azimuth pattern
DAS_azi = np.arctan2(np.diff(DAS_lat_Ridgecrest), np.diff(DAS_lon_Ridgecrest))
DAS_azi = np.concatenate((DAS_azi, [np.pi]))
DAS_azi = DAS_azi - np.pi

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

ieq = 19
for ieq in range(len(event_lon_Ridgecrest)):
    eq_lon = event_lon_Ridgecrest[ieq]
    eq_lat = event_lat_Ridgecrest[ieq]
    eq_id = event_id[ieq]

    eq_azi = np.arctan2((eq_lat - DAS_lat_Ridgecrest), (eq_lon - DAS_lon_Ridgecrest))
    eq_DAZ_incident = eq_azi - DAS_azi

    peak_amplitude_df_eq = peak_amplitude_df[peak_amplitude_df.event_id == eq_id]
    station_index = [np.where(idx == DAS_index)[0][0] for idx in peak_amplitude_df_eq.channel_id]

    eq_DAZ_incident = eq_DAZ_incident[station_index]

    pclip=95
    clipVal = np.percentile(np.absolute(peak_amplitude_df_eq.peak_P), pclip)

    ax.plot(eq_DAZ_incident, peak_amplitude_df_eq.peak_P/clipVal, 'o', alpha=0.01)#, markeredgecolor='k')

ax.set_rmax(1)
plt.figure()
plt.plot(eq_lon, eq_lat, 'rx')
plt.plot(DAS_lon_Ridgecrest, DAS_lat_Ridgecrest, '-')

#%%
# Looking into the amplitude pattern
DAS_azi = np.arctan2(np.diff(DAS_lat_Ridgecrest), np.diff(DAS_lon_Ridgecrest))
DAS_azi = np.concatenate((DAS_azi, [np.pi]))
DAS_azi = DAS_azi - np.pi

mean_peak_P = []
std_peak_P = []
for ieq in range(len(event_lon_Ridgecrest)):
    eq_id = event_id[ieq]
    peak_amplitude_df_eq = peak_amplitude_df[peak_amplitude_df.event_id == eq_id]

    peak_amplitude_df_eq.drop(peak_amplitude_df_eq[peak_amplitude_df_eq.peak_P > 10 * peak_amplitude_df_eq.peak_P.mean()].index, inplace=True)

    mean_peak_P.append(peak_amplitude_df_eq.peak_P.mean())
    std_peak_P.append(peak_amplitude_df_eq.peak_P.std())

fig, ax = plt.subplots(figsize=(12 ,8))
ax.errorbar(np.arange(len(event_lon_Ridgecrest)), mean_peak_P, yerr=std_peak_P, marker='s', linestyle='') 
ax.set_ylim(-1e3, 10e3)

# %%
# ==============================  Mammoth data ========================================
#%% Specify the file names
DAS_info_files1 = '/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Mammoth_South.txt'
DAS_info_files2 = '/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Mammoth_North.txt'
catalog_file =  '/home/yinjx/kuafu/Mammoth/catalog_regional.csv'
results_output_dir = '/home/yinjx/kuafu/Mammoth/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_name1 = '/South/Mammoth_South_Scaling_M3.csv'
das_pick_file_name2 = '/North/Mammoth_North_Scaling_M3.csv'

DAS_info_Mammoth_S = pd.read_csv(DAS_info_files1, sep=',', engine='python').dropna()
DAS_info_Mammoth_N = pd.read_csv(DAS_info_files2, sep=',', engine='python').dropna()
DAS_info = pd.concat([DAS_info_Mammoth_S, DAS_info_Mammoth_N], axis=0)

DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['channel'].astype('int')
DAS_lat_Mammoth = DAS_info['latitude']
DAS_lon_Mammoth = DAS_info['longitude']

peak_amplitude_df1 = pd.read_csv(results_output_dir + '/' + das_pick_file_name1)
peak_amplitude_df2 = pd.read_csv(results_output_dir + '/' + das_pick_file_name2)
peak_amplitude_df_Mammoth = pd.concat([peak_amplitude_df1, peak_amplitude_df2], axis=0, ignore_index=True)

catalog_Mammoth = pd.read_csv(catalog_file, sep=',', engine='python')
# Find events in the pick file
event_id_selected_Mammoth = np.unique(peak_amplitude_df_Mammoth['event_id'])
catalog_select_Mammoth = catalog_Mammoth[catalog_Mammoth['ID'].isin(event_id_selected_Mammoth)]
num_events_Mammoth = catalog_select_Mammoth.shape[0]
catalog_select_Mammoth

# Add the event label for plotting
peak_amplitude_df = add_event_label(peak_amplitude_df_Mammoth)

#%%
# plot time variation of events
time_list_Mammoth = [obspy.UTCDateTime(time.replace(' ', 'T')) for time in catalog_select_Mammoth['time']]
time_span_Mammoth = np.array([time-time_list_Mammoth[0] for time in time_list_Mammoth])
time_span_days_Mammoth = time_span_Mammoth/3600/24

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(time_span_days_Mammoth, catalog_select_Mammoth['mag'], 'o')
ax.set_xlabel(f'Days from the {time_list_Mammoth[0]}')
ax.set_ylabel('magnitude')
plt.savefig(results_output_dir + '/time_variation_selected_earthquakes.png', bbox_inches='tight')

# plot map
fig, ax = plt.subplots(figsize=(7, 6))
cmp = ax.scatter(DAS_lon_Mammoth, DAS_lat_Mammoth, s=10, c=DAS_index, cmap='jet')
ax.scatter(catalog_select_Mammoth['lon'], catalog_select_Mammoth['lat'], s=10**(catalog_select_Mammoth['mag']/5), c='k')
ax.set_title(f'Total event number: {num_events_Mammoth}')
fig.colorbar(cmp)
plt.savefig(results_output_dir + '/map_of_earthquakes_not_grouped.png', bbox_inches='tight')

# plot data statistics
peak_amplitude_df_temp = peak_amplitude_df.iloc[::5, :]
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km)
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P)
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S
plt.figure(figsize=(14,8))
sns.pairplot(peak_amplitude_df_temp[['magnitude','log10(distance)', 'log10(peak_P)','P/S']], kind='hist', diag_kind="kde", corner=True)
plt.savefig(results_output_dir + '/mammoth_data_statistics.png')

#%%
# Define region of interest around Ridgecrest
region = [-120, -117.5, 36.5, 39]
projection = "M12c"
# Load sample grid (3 arc second global relief) in target area
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=region)

# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=15)

# --------------- plotting the original Data Elevation Model -----------

fig.basemap(region=region, 
projection=projection, 
frame=['WSrt+t"Long Valley"', "xa0.5", "ya0.5"]
)
pygmt.makecpt(cmap="dem4", series=[-1000, 3000])
fig.grdimage(
    grid=grid,
    projection=projection,
    cmap=True,
    shading='+a45+nt1',
    transparency=40
)

fig.plot(x=catalog_select_Mammoth['lon'], y=catalog_select_Mammoth['lat'], style="c0.15c", color="black")

fig.plot(x=DAS_lon_Mammoth[::10], y=DAS_lat_Mammoth[::10], style="c0.05c", color="red")

fig.show()
fig.savefig(results_output_dir + '/map_of_earthquakes_GMT_Mammoth.png')


#%%
# ================================ Work on both events  =======================================
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'

time_span_Mammoth_ref = np.array([time-time_list_Ridgecrest[0] for time in time_list_Mammoth])
time_span_days_Mammoth_ref = time_span_Mammoth_ref/3600/24

fig, ax = plt.subplots(figsize=(6, 14))
ax.plot(catalog_select_Ridgecrest[7], time_span_days_Ridgecrest, 'ro', label='Ridgecrest earthquakes')
ax.plot(catalog_select_Mammoth['mag'], time_span_days_Mammoth_ref, 'bo', label='Mammoth earthquakes')

ax.invert_yaxis()
ax.set_ylim(10, 880)
ax.set_ylabel(f'Days from the {time_list_Ridgecrest[0].isoformat()[:10]}')
ax.set_xlabel('magnitude')
ax.grid()
ax.legend(loc=1)
plt.savefig(results_output_dir + '/time_variation_selected_earthquakes.png', bbox_inches='tight')


#%%
# Plot both arrays
# Define region of interest around Ridgecrest
region = [-120, -117, 35, 39]
projection = "M12c"
# Load sample grid (3 arc second global relief) in target area
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=region)

# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=15)

# --------------- plotting the original Data Elevation Model -----------

fig.basemap(region=region, 
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

fig.plot(x=catalog_select_Mammoth['lon'], y=catalog_select_Mammoth['lat'], style="c0.1c", color="black")
fig.plot(x=event_lon_Ridgecrest, y=event_lat_Ridgecrest, style="c0.1c", color="black")

fig.plot(x=DAS_info_Mammoth_S['longitude'][::10], y=DAS_info_Mammoth_S['latitude'][::10], style="c0.1c", color="blue")
fig.plot(x=DAS_info_Mammoth_N['longitude'][::10], y=DAS_info_Mammoth_N['latitude'][::10], style="c0.1c", color="blue")
#fig.plot(x=DAS_lon_Mammoth[::10], y=DAS_lat_Mammoth[::10], style="c0.1c", color="red")
fig.plot(x=DAS_lon_Ridgecrest[::10], y=DAS_lat_Ridgecrest[::10], style="c0.1c", color="blue")

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
    rectangle = [[region[0], region[2], region[1], region[3]]]
    fig.plot(data=rectangle, style="r+s", pen="2p,red")

fig.show()
fig.savefig(results_output_dir + '/map_of_earthquakes.png')





# %%
# # ==============================  Olancha data ========================================
# #%% Specify the file names
# DAS_info_files = '/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Olancha_Plexus.txt'
# catalog_file =  '/home/yinjx/notebooks/strain_scaling/Ridgecrest_das_catalog_M2_M8.txt'
# results_output_dir = '/kuafu/yinjx/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain_rate'
# das_pick_file_name = '/peak_amplitude_M3+.csv'
# peak_df_file = results_output_dir + '/' + das_pick_file_name
# # Load the peak amplitude results
# peak_amplitude_df = pd.read_csv(peak_df_file)

# #  load the information about the Olancha DAS channel (needs to further update)
# DAS_info = np.genfromtxt(DAS_info_files)
# DAS_channel_num = DAS_info.shape[0]
# DAS_index = DAS_info[:, 0].astype('int')
# DAS_lon = DAS_info[:, 1]
# DAS_lat = DAS_info[:, 2]

# %% This works for different regions
# Show the peak strain rate amplitude variations
plot_magnitude_distance_coverage(peak_amplitude_df, results_output_dir + '/magnitude_distance_distribution.png')
plot_distance_variations(peak_amplitude_df, ['peak_P', 'peak_S'], 1e4, results_output_dir + '/peak_strain_rate_vs_distance.png')
plot_magnitude_variations(peak_amplitude_df, ['peak_P', 'peak_S'], 1e4, results_output_dir + '/peak_strain_rate_vs_magnitude.png')
plot_P_S_amplitude_ratio(peak_amplitude_df, ['peak_P', 'peak_S'], results_output_dir + '/peak_strain_rate_P_S_ratio.png')
plot_peak_time_distance_variations(peak_amplitude_df, results_output_dir + '/peak_amplitude_time.png')
# %% 
# Show the peak strain amplitude variations
plot_distance_variations(peak_amplitude_df, ['peak_P_strain', 'peak_S_strain'], results_output_dir + '/peak_strain_vs_distance.png')
plot_magnitude_variations(peak_amplitude_df, ['peak_P_strain', 'peak_S_strain'], results_output_dir + '/peak_strain_vs_magnitude.png')
plot_P_S_amplitude_ratio(peak_amplitude_df, ['peak_P_strain', 'peak_S_strain'], results_output_dir + '/peak_strain_P_S_ratio.png')
# %%


def plot_peak_time_distance_variations(peak_amplitude_df,ymax, figure_name):
    ''' Look at the rough variation of measured peak time with distance'''
    fig, ax = plt.subplots(2, 1, figsize=(10, 20), sharex=True, sharey=True)
    ax[0].scatter(peak_amplitude_df.distance_in_km, peak_amplitude_df.max_P_time, 
              s=10, c=peak_amplitude_df.event_label, marker='o', cmap='jet', alpha=0.01)
    ax[0].plot(peak_amplitude_df.distance_in_km, peak_amplitude_df.max_P_time2 + peak_amplitude_df.max_P_time, 'k.',markersize=0.1)
    
    ax[1].scatter(peak_amplitude_df.distance_in_km, peak_amplitude_df.max_S_time, 
              s=20, c=peak_amplitude_df.event_label, marker='x', cmap='jet', alpha=0.1)

    #ax[1].set_xscale('log')


    ax[0].set_ylabel('Peak P amplitude time after P')
    ax[0].set_xlabel('Distance (km)')
    ax[0].set_title('(a) Maximum P time')
    ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
    ax[1].set_title('(b) Maximum S time')
    ax[1].set_xlabel('Distance (km))')
    ax[1].set_ylabel('Peak S amplitude time after S')
    ax[1].set_ylim(top=ymax)

    plt.savefig(figure_name, bbox_inches='tight')

plot_peak_time_distance_variations(peak_amplitude_df, 20, results_output_dir + '/peak_amplitude_time.png')
# %%
