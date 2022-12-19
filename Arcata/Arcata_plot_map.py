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

import pygmt


#%% 
# Specify the file names
combined_results_output_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate'
figure_output_dir = combined_results_output_dir + '/data_figures'
mkdir(figure_output_dir)

event_folder_list = ['/kuafu/EventData/Arcata_Spring2022']

region_label = ['arcata']
region_legend_text = ['Arcata'] 

snr_threshold = 10
min_channel = 100
magnitude_threshold = [2, 10]

snr_threshold_Sanriku = 5
#%% 
# Combined all data 
peak_amplitude_df_all = pd.read_csv(combined_results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')

# use the hypocentral distance instead of epicentral distance
peak_amplitude_df_all['distance_in_km'] = peak_amplitude_df_all['calibrated_distance_in_km']

# some event filtering
peak_amplitude_df_all = peak_amplitude_df_all[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km', 
                                    'snrP', 'snrS', 'peak_P', 'peak_S', 'region']] #, 'combined_channel_id', 'event_label', 'region_site'
# to remove some extreme values
peak_amplitude_df_all = remove_outliers(peak_amplitude_df_all, outlier_value=1e4)
peak_amplitude_df_all = peak_amplitude_df_all.drop(index=peak_amplitude_df_all[peak_amplitude_df_all.peak_P<=0].index)
peak_amplitude_df_all = peak_amplitude_df_all.drop(index=peak_amplitude_df_all[peak_amplitude_df_all.peak_S<=0].index)

# events to show
event_id_selected = np.unique(peak_amplitude_df_all['event_id'])

DAS_info_all = pd.DataFrame(columns=['index', 'latitude', 'longitude', 'elevation_m'])
catalog_select_all = pd.DataFrame(columns=['event_id', 'event_time', 'longitude', 'latitude','depth_km','magnitude','magnitude_type','source', 'region'])
for i_region in range(len(event_folder_list)):
    event_folder = event_folder_list[i_region]

    DAS_info_files = event_folder + '/das_info.csv'
    catalog_file =  event_folder + '/catalog.csv'

    #  load the information about the DAS channel
    DAS_info = pd.read_csv(DAS_info_files)
    DAS_info_all = pd.concat((DAS_info_all, DAS_info), axis=0, ignore_index=True)

    catalog = pd.read_csv(catalog_file)
    # Find events in the pick file
    catalog_select = catalog
    # catalog_select = catalog[catalog.event_id.isin(event_id_selected)]
    catalog_select['region'] = region_label[i_region]
    catalog_select_all = pd.concat((catalog_select_all, catalog_select), axis=0, ignore_index=True)

DAS_channel_num = DAS_info_all.shape[0]
DAS_index = DAS_info_all['index'].astype('int')
DAS_lon = DAS_info_all.longitude
DAS_lat = DAS_info_all.latitude

# specified events
specified_event_list = [73736021, 73739276, 73747016, 73751651, 73757961, 73758756, 
73735891, 73741131, 73747621, 73747806, 73748011, 73753546, 73743421] #73739346, 


catalog_select_all = catalog_select_all[catalog_select_all.event_id.isin(specified_event_list)]

num_events = catalog_select_all.shape[0]
event_lon = catalog_select_all.longitude
event_lat = catalog_select_all.latitude
event_id = catalog_select_all.event_id


#%%
# =========================  Plot both arrays in California with PyGMT ==============================
gmt_region = [-125.5, -123, 39.5, 41.5]

projection = "M12c"
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=gmt_region)

# calculate the reflection of a light source projecting from west to east
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=15, FONT_TITLE="20p,Helvetica,black")

# --------------- plotting the original Data Elevation Model -----------
fig.basemap(region=gmt_region, 
projection=projection, 
frame=['WSrt', "xa1.0", "ya1.0"] #'WSrt+t"Arcata DAS experiment"'
)
pygmt.makecpt(cmap="geo", series=[-4000, 4000])
fig.grdimage(
    grid=grid,
    projection=projection,
    cmap=True,
    shading='+a45+nt1',
    transparency=35
)

fig.plot(x=catalog_select_all.longitude.astype('float'), y=catalog_select_all.latitude.astype('float'), style="c0.2c", color="black")
fig.plot(x=DAS_info_all.longitude[::10].astype('float'), y=DAS_info_all.latitude[::10].astype('float'), style="c0.05c", color="red")
fig.text(text="Arcata array", x=-124, y=40.93, font="12p,Helvetica-Bold,red")


# special_event = catalog_select_all[catalog_select_all.event_id == 73743421]
# x_73743421 = special_event.longitude.astype('float') 
# y_73743421 = special_event.latitude.astype('float')
# fig.plot(x=x_73743421, y=y_73743421, style="c0.3c", color="blue")
# fig.text(text=f"id {special_event.iloc[0, :].event_id}, M {special_event.iloc[0, :].magnitude}", x=x_73743421, y=y_73743421-0.1, font="12p,Helvetica-Bold,blue")

# # Create an inset map
# with fig.inset(position="jBL+w4.3c/7.5c+o0.1c", box="+gwhite+p1p"):
#     fig.coast(
#         region=[-125, -115, 30, 45],
#         projection="J",
#         #land='gray',#
#         dcw="US+ggray+p0.2p",
#         #water='lightblue',
#         area_thresh=10000,
#         resolution='f',
#         borders=["1/0.5p,black", "2/0.25p,black"]
#     )
#     # Plot a rectangle ("r") in the inset map to show the area of the main
#     # figure. 
#     rectangle = [[gmt_region[0], gmt_region[2], gmt_region[1], gmt_region[3]]]
#     fig.plot(data=rectangle, style="r+s", pen="2p,red")

fig.show()
fig.savefig(figure_output_dir + '/map_of_earthquakes_Arcata_GMT_0.png')



# %%
# =================================   New organization of data statistics (3 by 3) Contour plot =================================  
# =================================   Merge events  =================================  
levels = [0.05, 0.3, 0.6, 0.9]
linewidths = [0.8, 1.5, 2.5, 3.5]
thresh = 0.01

bins = (10, 10)
add_possible_clipping = False
peak_amplitude_df_temp = peak_amplitude_df_all.iloc[::1, :]
peak_amplitude_df_temp = peak_amplitude_df_temp.groupby(by=['event_id', 'region'], as_index=False).median()

calibrated_distance = ''#'_calibrated_distance'
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
# peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P.astype('float'))
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float'))
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S

peak_amplitude_df_Sanriku = peak_amplitude_df_temp[peak_amplitude_df_temp.region == 'sanriku']
peak_amplitude_df_others = peak_amplitude_df_temp[peak_amplitude_df_temp.region != 'sanriku']

plt.close('all')
fig, ax = plt.subplots(3, 3, figsize=(14, 14))
gca = ax[0, 0]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['magnitude', 'region']], x='magnitude', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['magnitude', 'region']], x='magnitude', hue='region', palette='Set2', legend=False)
gca.set_xlim(1, 6)

gca = ax[0, 1]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)', 'region']], x='log10(distance)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)', 'region']], x='log10(distance)', hue='region', palette='Set2', legend=False)
gca.set_xlim(0, 3)

gca = ax[1, 0]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['magnitude','log10(peak_P)', 'region']], x='magnitude', y='log10(peak_P)',
                 hue='region', palette='Set2', alpha=1, cbar=False, legend=True, fill=False, levels=levels, thresh=thresh, linewidths=linewidths)
gca.set_ylim(-2.5, 2.5)
g.get_legend().set_bbox_to_anchor((3.5, 2.5))
# update the legend text
L = g.get_legend()
for i_L in range(len(L.get_texts())):
    L.get_texts()[i_L].set_text(region_legend_text[i_L])

gca = ax[1, 1]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_P)', 'region']], x='log10(distance)', y='log10(peak_P)',
                 hue='region', palette='Set2', alpha=1, cbar=False, legend=False, levels=levels, thresh=thresh, linewidths=linewidths)

gca = ax[2, 0]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['magnitude','log10(peak_S)', 'region']], x='magnitude', y='log10(peak_S)',
                 hue='region', palette='Set2', alpha=1, cbar=False, legend=False, levels=levels, thresh=thresh, linewidths=linewidths)
gca.plot(peak_amplitude_df_Sanriku.magnitude.astype('float'), np.log10(peak_amplitude_df_Sanriku.peak_S.astype('float')), 'o', markersize=3, color='#DA8EC0', alpha=1)
gca.set_ylim(-2.5, 2.5)


gca = ax[2, 1]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_S)', 'region']], x='log10(distance)', y='log10(peak_S)',
                 hue='region', palette='Set2', alpha=1, cbar=False, legend=False, levels=levels, thresh=thresh, linewidths=linewidths)
gca.plot(np.log10(peak_amplitude_df_Sanriku.distance_in_km.astype('float')), np.log10(peak_amplitude_df_Sanriku.peak_S.astype('float')), 'o', markersize=3, color='#DA8EC0', alpha=1)

fig.delaxes(ax[0, 2])

gca = ax[1, 2]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(peak_P)', 'region']], x='log10(peak_P)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(peak_P)', 'region']], x='log10(peak_P)', hue='region', palette='Set2', legend=False)
gca.set_xlim(-2.5, 2.5)

gca = ax[2, 2]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(peak_S)', 'region']], x='log10(peak_S)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(peak_S)', 'region']], x='log10(peak_S)', hue='region', palette='Set2', legend=False)

# share x axis
ax[1, 0].sharex(ax[0, 0])
ax[2, 0].sharex(ax[0, 0])
ax[1, 1].sharex(ax[0, 1])
ax[2, 1].sharex(ax[0, 1])
ax[2, 2].sharex(ax[1, 2])

# share y axis
ax[1, 1].sharey(ax[1, 0])
ax[2, 1].sharey(ax[2, 0])
ax[2, 2].sharey(ax[1, 2])

# some final handling
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for i_ax, gca in enumerate(ax.flatten()):
    gca.spines.right.set_visible(False)
    gca.spines.top.set_visible(False)
    # add annotation
    if i_ax != 2:
        gca.annotate(f'({letter_list[k]})', xy=(-0.3, 1.1), xycoords=gca.transAxes)
        k += 1
    # remove some labels
    if i_ax not in [0, 3, 6, 5, 8]:
        gca.set_ylabel('')
    if i_ax not in [5, 6, 7, 8]:
        gca.set_xlabel('')

plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.savefig(figure_output_dir + f'/data_correlation{calibrated_distance}_3x3_contours.png', bbox_inches='tight')

# %%
# =================================   New organization of data statistics (3 by 3) hist plot =================================  
# =================================   Merge events  =================================  
def adjust_region_order(peak_amplitude_df, region_keys):
    for ii, region_key in enumerate(region_keys):
        if ii == 0:
            peak_amplitude_df_temp = peak_amplitude_df[peak_amplitude_df.region == region_key]
        else:
            peak_amplitude_df_temp = pd.concat([peak_amplitude_df_temp, peak_amplitude_df[peak_amplitude_df.region == region_key]], axis=0)
    return peak_amplitude_df_temp
    
levels = 1
thresh = 0.05

bins = (40, 40)
add_possible_clipping = False
peak_amplitude_df_temp = peak_amplitude_df_all.iloc[::1, :]
peak_amplitude_df_temp = peak_amplitude_df_temp.groupby(by=['event_id', 'region'], as_index=False).median()

peak_amplitude_df_temp = adjust_region_order(peak_amplitude_df_temp, ['ridgecrest', 'mammothN', 'mammothS', 'sanriku'])


calibrated_distance = ''#'_calibrated_distance'
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
# peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P.astype('float'))
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float'))
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S

peak_amplitude_df_Sanriku = peak_amplitude_df_temp[peak_amplitude_df_temp.region == 'sanriku']
peak_amplitude_df_others = peak_amplitude_df_temp[peak_amplitude_df_temp.region != 'sanriku']

plt.close('all')
fig, ax = plt.subplots(3, 3, figsize=(14, 14))
gca = ax[0, 0]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['magnitude', 'region']], x='magnitude', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['magnitude', 'region']], x='magnitude', hue='region', palette='Set2', legend=False)
gca.set_xlim(1, 6)

gca = ax[0, 1]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(distance)', 'region']], x='log10(distance)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(distance)', 'region']], x='log10(distance)', hue='region', palette='Set2', legend=False)
gca.set_xlim(0, 3)

gca = ax[1, 0]
# g = sns.scatterplot(ax=gca, data=peak_amplitude_df_temp[['magnitude','log10(peak_P)', 'region']], x='magnitude', y='log10(peak_P)', size=1,
#                  hue='region', palette='Set2', alpha=0.2, legend=False)
g = sns.histplot(ax=gca, data=peak_amplitude_df_temp[['magnitude','log10(peak_P)', 'region']], x='magnitude', y='log10(peak_P)',
                 color=".1", alpha=0.4, pmax=0.5, cbar=True, bins=bins, legend=False, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0, 'label':'Counts'})
g = sns.histplot(ax=gca, data=peak_amplitude_df_temp[['magnitude','log10(peak_P)', 'region']], x='magnitude', y='log10(peak_P)',
                 hue='region', palette='Set2', alpha=0.4, pmax=0.5, cbar=False, bins=bins, legend=True, fill=False)
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['magnitude','log10(peak_P)', 'region']], x='magnitude', y='log10(peak_P)',
                 hue='region', palette='Set2', alpha=1, pmax=0.5, cbar=False, bins=bins, legend=False, fill=False, levels=levels, thresh=thresh)

gca.set_ylim(-2.5, 2.5)
g.get_legend().set_bbox_to_anchor((3.5, 2.5))
# update the legend text
L = g.get_legend()
for i_L in range(len(L.get_texts())):
    L.get_texts()[i_L].set_text(region_legend_text[i_L])

gca = ax[1, 1]
g = sns.histplot(ax=gca, data=peak_amplitude_df_temp[['log10(distance)','log10(peak_P)', 'region']], x='log10(distance)', y='log10(peak_P)',
                 color=".1", alpha=0.4, pmax=0.9, cbar=True, bins=bins, legend=False, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0, 'label':'Counts'})
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_P)', 'region']], x='log10(distance)', y='log10(peak_P)',
                 hue='region', palette='Set2', alpha=0.4, pmax=0.9, cbar=False, bins=bins, legend=False)
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_P)', 'region']], x='log10(distance)', y='log10(peak_P)',
                 hue='region', palette='Set2', alpha=1, pmax=0.9, cbar=False, bins=bins, legend=False, levels=levels, thresh=thresh)

gca = ax[2, 0]
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['magnitude','log10(peak_S)', 'region']], x='magnitude', y='log10(peak_S)',
                 color=".1", alpha=0.4, pmax=0.5, cbar=True, bins=bins, legend=False, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0, 'label':'Counts'})
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['magnitude','log10(peak_S)', 'region']], x='magnitude', y='log10(peak_S)',
                 hue='region', palette='Set2', alpha=0.4, pmax=0.5, cbar=False, bins=bins, legend=False)
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['magnitude','log10(peak_S)', 'region']], x='magnitude', y='log10(peak_S)',
                 hue='region', palette='Set2', alpha=1, pmax=0.5, cbar=False, bins=bins, legend=False, levels=levels, thresh=thresh)

gca.plot(peak_amplitude_df_Sanriku.magnitude.astype('float'), np.log10(peak_amplitude_df_Sanriku.peak_S.astype('float')), 'o', markersize=3, color='#DA8EC0', alpha=1)
gca.set_ylim(-2.5, 2.5)


gca = ax[2, 1]
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_S)', 'region']], x='log10(distance)', y='log10(peak_S)',
                 color=".1", alpha=0.4, pmax=0.9, cbar=True, bins=bins, legend=False, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0, 'label':'Counts'})
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_S)', 'region']], x='log10(distance)', y='log10(peak_S)',
                 hue='region', palette='Set2', alpha=0.4, pmax=0.9, cbar=False, bins=bins, legend=False)
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_S)', 'region']], x='log10(distance)', y='log10(peak_S)',
                 hue='region', palette='Set2', alpha=1, pmax=0.5, cbar=False, bins=bins, legend=False, levels=levels, thresh=thresh)

gca.plot(np.log10(peak_amplitude_df_Sanriku.distance_in_km.astype('float')), np.log10(peak_amplitude_df_Sanriku.peak_S.astype('float')), 'o', markersize=3, color='#DA8EC0', alpha=1)

fig.delaxes(ax[0, 2])

gca = ax[1, 2]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(peak_P)', 'region']], x='log10(peak_P)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(peak_P)', 'region']], x='log10(peak_P)', hue='region', palette='Set2', legend=False)
gca.set_xlim(-2.5, 2.5)

gca = ax[2, 2]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(peak_S)', 'region']], x='log10(peak_S)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(peak_S)', 'region']], x='log10(peak_S)', hue='region', palette='Set2', legend=False)

# share x axis
ax[1, 0].sharex(ax[0, 0])
ax[2, 0].sharex(ax[0, 0])
ax[1, 1].sharex(ax[0, 1])
ax[2, 1].sharex(ax[0, 1])
ax[2, 2].sharex(ax[1, 2])

# share y axis
ax[1, 1].sharey(ax[1, 0])
ax[2, 1].sharey(ax[2, 0])
ax[2, 2].sharey(ax[1, 2])

# some final handling
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for i_ax, gca in enumerate(ax.flatten()):
    gca.spines.right.set_visible(False)
    gca.spines.top.set_visible(False)
    # add annotation
    if i_ax != 2:
        gca.annotate(f'({letter_list[k]})', xy=(-0.3, 1.1), xycoords=gca.transAxes)
        k += 1
    # remove some labels
    if i_ax not in [0, 3, 6, 5, 8]:
        gca.set_ylabel('')
    if i_ax not in [5, 6, 7, 8]:
        gca.set_xlabel('')

plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.savefig(figure_output_dir + f'/data_correlation{calibrated_distance}_3x3_hist.png', bbox_inches='tight')

#%% 
#  =================================  plot data statistics (peak strain rate measurements) =================================  
peak_amplitude_df_temp = peak_amplitude_df_all.iloc[::1, :]
calibrated_distance = ''#'_calibrated_distance'
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
# peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P.astype('float'))
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float'))
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S


fig = plt.figure(figsize=(14,8))

g = sns.PairGrid(peak_amplitude_df_temp[['magnitude','log10(distance)', 'log10(peak_P)','log10(peak_S)', 'region']], 
                 diag_sharey=False, corner=True, hue='region', palette='Set2', height=4, aspect=0.8)


g.map_diag(sns.kdeplot, hue=None, color=".1", linewidth=2) #hue='Array', palette='Set2', 
g.map_diag(sns.kdeplot, linewidth=2, alpha=1) #hue='Array', palette='Set2', 

g.map_lower(sns.histplot, pmax=0.5, hue=None, color=".1", cbar=True, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0}, bins=80)
g.map_lower(sns.histplot, pmax=0.5, alpha=0.3, bins=80)

plt.subplots_adjust(hspace=0.2, wspace=0.25)

g.add_legend()
g._legend.set_bbox_to_anchor((0.3, 0.9))

# show the Sanriku data
peak_amplitude_df_Sanriku = peak_amplitude_df_temp[peak_amplitude_df_temp.region == 'sanriku']
g.axes[1,0].plot(peak_amplitude_df_Sanriku.magnitude, np.log10(peak_amplitude_df_Sanriku.distance_in_km), '.', markersize=1, color='#DA8EC0', alpha=0.3)
g.axes[3,0].plot(peak_amplitude_df_Sanriku.magnitude, np.log10(peak_amplitude_df_Sanriku.peak_S), '.', markersize=1, color='#DA8EC0', alpha=0.3)
g.axes[3,1].plot(np.log10(peak_amplitude_df_Sanriku.distance_in_km), np.log10(peak_amplitude_df_Sanriku.peak_S), '.', markersize=1, color='#DA8EC0', alpha=0.3)

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
plt.savefig(figure_output_dir + f'/data_correlation{calibrated_distance}.png', bbox_inches='tight')
plt.savefig(f'/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/data_correlation{calibrated_distance}.png', bbox_inches='tight')

#%% 
# =================================   New organization of data statistics (3 by 3) (peak strain rate measurements) =================================  
bins = (100, 100)
add_possible_clipping = False
peak_amplitude_df_temp = peak_amplitude_df_all.iloc[::1, :]
calibrated_distance = ''#'_calibrated_distance'
peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
# peak_amplitude_df_temp['log10(distance)'] = np.log10(peak_amplitude_df_temp.distance_in_km.astype('float'))
peak_amplitude_df_temp['log10(peak_P)'] = np.log10(peak_amplitude_df_temp.peak_P.astype('float'))
peak_amplitude_df_temp['log10(peak_S)'] = np.log10(peak_amplitude_df_temp.peak_S.astype('float'))
peak_amplitude_df_temp['P/S'] = peak_amplitude_df_temp.peak_P/peak_amplitude_df_temp.peak_S

peak_amplitude_df_Sanriku = peak_amplitude_df_temp[peak_amplitude_df_temp.region == 'sanriku']
peak_amplitude_df_others = peak_amplitude_df_temp[peak_amplitude_df_temp.region != 'sanriku']

plt.close('all')
fig, ax = plt.subplots(3, 3, figsize=(14, 14))
gca = ax[0, 0]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['magnitude', 'region']], x='magnitude', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['magnitude', 'region']], x='magnitude', hue='region', palette='Set2', legend=False)
gca.set_xlim(1, 6)

gca = ax[0, 1]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(distance)', 'region']], x='log10(distance)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(distance)', 'region']], x='log10(distance)', hue='region', palette='Set2', legend=False)
gca.set_xlim(0, 3)

gca = ax[1, 0]
g = sns.histplot(ax=gca, data=peak_amplitude_df_temp[['magnitude','log10(peak_P)', 'region']], x='magnitude', y='log10(peak_P)',
                 color=".1", pmax=0.5, cbar=True, bins=bins, legend=False, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0, 'label':'Counts'})
g = sns.histplot(ax=gca, data=peak_amplitude_df_temp[['magnitude','log10(peak_P)', 'region']], x='magnitude', y='log10(peak_P)',
                 hue='region', palette='Set2', alpha=0.3, pmax=0.5, cbar=False, bins=bins, legend=True)
gca.set_ylim(-2.5, 2.5)
g.get_legend().set_bbox_to_anchor((3.5, 2.5))
# update the legend text
L = g.get_legend()
for i_L in range(len(L.get_texts())):
    L.get_texts()[i_L].set_text(region_legend_text[i_L])

gca = ax[1, 1]
g = sns.histplot(ax=gca, data=peak_amplitude_df_temp[['log10(distance)','log10(peak_P)', 'region']], x='log10(distance)', y='log10(peak_P)',
                 color=".1", pmax=0.5, cbar=True, bins=bins, legend=False, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0, 'label':'Counts'})
g = sns.histplot(ax=gca, data=peak_amplitude_df_temp[['log10(distance)','log10(peak_P)', 'region']], x='log10(distance)', y='log10(peak_P)',
                 hue='region', palette='Set2', alpha=0.3, pmax=0.5, cbar=False, bins=bins, legend=False)

gca = ax[2, 0]
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['magnitude','log10(peak_S)', 'region']], x='magnitude', y='log10(peak_S)',
                 color=".1", pmax=0.5, cbar=True, bins=bins, legend=False, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0, 'label':'Counts'})
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['magnitude','log10(peak_S)', 'region']], x='magnitude', y='log10(peak_S)',
                 hue='region', palette='Set2', alpha=0.3, pmax=0.5, cbar=False, bins=bins, legend=False)
gca.plot(peak_amplitude_df_Sanriku.magnitude.astype('float'), np.log10(peak_amplitude_df_Sanriku.peak_S.astype('float')), 'o', markersize=1, color='#DA8EC0', alpha=0.4)
gca.set_ylim(-2.5, 2.5)


gca = ax[2, 1]
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_S)', 'region']], x='log10(distance)', y='log10(peak_S)',
                 color=".1", pmax=0.5, cbar=True, bins=bins, legend=False, cbar_kws={'location': 'top', 'fraction':0.02, 'pad':0, 'label':'Counts'})
g = sns.histplot(ax=gca, data=peak_amplitude_df_others[['log10(distance)','log10(peak_S)', 'region']], x='log10(distance)', y='log10(peak_S)',
                 hue='region', palette='Set2', alpha=0.3, pmax=0.5, cbar=False, bins=bins, legend=False)
gca.plot(np.log10(peak_amplitude_df_Sanriku.distance_in_km.astype('float')), np.log10(peak_amplitude_df_Sanriku.peak_S.astype('float')), 'o', markersize=1, color='#DA8EC0', alpha=0.4)

fig.delaxes(ax[0, 2])

gca = ax[1, 2]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(peak_P)', 'region']], x='log10(peak_P)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(peak_P)', 'region']], x='log10(peak_P)', hue='region', palette='Set2', legend=False)
gca.set_xlim(-2.5, 2.5)

gca = ax[2, 2]
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(peak_S)', 'region']], x='log10(peak_S)', color='k')
g = sns.kdeplot(ax=gca, data=peak_amplitude_df_temp[['log10(peak_S)', 'region']], x='log10(peak_S)', hue='region', palette='Set2', legend=False)

# share x axis
ax[1, 0].sharex(ax[0, 0])
ax[2, 0].sharex(ax[0, 0])
ax[1, 1].sharex(ax[0, 1])
ax[2, 1].sharex(ax[0, 1])
ax[2, 2].sharex(ax[1, 2])

# share y axis
ax[1, 1].sharey(ax[1, 0])
ax[2, 1].sharey(ax[2, 0])
ax[2, 2].sharey(ax[1, 2])

# some final handling
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for i_ax, gca in enumerate(ax.flatten()):
    gca.spines.right.set_visible(False)
    gca.spines.top.set_visible(False)
    # add annotation
    if i_ax != 2:
        gca.annotate(f'({letter_list[k]})', xy=(-0.3, 1.1), xycoords=gca.transAxes)
        k += 1
    # remove some labels
    if i_ax not in [0, 3, 6, 5, 8]:
        gca.set_ylabel('')
    if i_ax not in [5, 6, 7, 8]:
        gca.set_xlabel('')

plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.savefig(figure_output_dir + f'/data_correlation{calibrated_distance}_3x3.png', bbox_inches='tight')
plt.savefig(f'/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/data_correlation{calibrated_distance}_3x3.png', bbox_inches='tight')




# %%
