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
combined_results_output_dir = '/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate'
figure_output_dir = combined_results_output_dir + '/data_figures'
mkdir(figure_output_dir)

event_folder = '/kuafu/EventData/Curie'

region_label = ['curie']
region_legend_text = ['Curie'] 

snr_threshold = 10
min_channel = 100
magnitude_threshold = [2, 10]

#%% 
DAS_info = pd.read_csv(event_folder + '/das_info.csv')
catalog = pd.read_csv(event_folder + '/catalog.csv')


# events to show
event_id_selected = [9001, 9007, 9006]


DAS_channel_num = DAS_info.shape[0]
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

# specified events
specified_event_list = [9006, 9007, 9006] #73739346, 
catalog_select_all = catalog[catalog.event_id.isin(event_id_selected)]

num_events = catalog_select_all.shape[0]
event_lon = catalog_select_all.longitude
event_lat = catalog_select_all.latitude
event_id = catalog_select_all.event_id

# load the regional permanent station
stations = pd.read_csv('/kuafu/yinjx/Curie/curie_nearby_stations.csv', index_col=None)
moment_tensor_catalog = pd.read_csv('/kuafu/yinjx/Curie/moment_tensor_catalog.csv', index_col=None, header=None)

plt.hist(moment_tensor_catalog[2])
plt.xlabel('depth (km)')
plt.ylabel('Counts')
plt.title(f'mean: {moment_tensor_catalog[2].mean():.2f}km, median: {moment_tensor_catalog[2].median():.2f}km')


#%%
# =========================  Plot both arrays in Chile with PyGMT ==============================
plt.close('all')
gmt_region = [-72.5, -70.9, -33.3, -31.8]

projection = "M12c"
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=gmt_region)

# calculate the reflection of a light source projecting from west to east
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=12, FONT_TITLE="14p,Helvetica,black")

# --------------- plotting the original Data Elevation Model -----------
fig.basemap(region=gmt_region, 
projection=projection, 
frame=['WSrt', "xa0.5", "ya0.5"] #'WSrt+t"Arcata DAS experiment"'
)
pygmt.makecpt(cmap="geo", series=[-4000, 4000])
fig.grdimage(
    grid=grid,
    projection=projection,
    cmap=True,
    shading='+a45+nt1',
    transparency=35
)

fig.plot(x=stations.Longitude.astype('float'), y=stations.Latitude.astype('float'), style="i0.8c", color="darkred")
# fig.plot(x=stations[stations.Station=='VA01'].Longitude.astype('float'), 
#          y=stations[stations.Station=='VA01'].Latitude.astype('float'), 
#          style="i0.8c", pen="1p,white", color="darkred")

fig.plot(x=catalog_select_all.longitude.astype('float'), y=catalog_select_all.latitude.astype('float'), style="c0.3c", color="black")
for ii in range(catalog_select_all.shape[0]):
    fig.text(text=catalog_select_all.iloc[ii, :].place, x=catalog_select_all.iloc[ii, :].longitude.astype('float'), 
        y=catalog_select_all.iloc[ii, :].latitude.astype('float')-0.05, font="10p,Helvetica-Bold,black")
fig.plot(x=DAS_info.longitude[::100].astype('float'), y=DAS_info.latitude[::100].astype('float'), style="c0.05c", color="red")
fig.text(text="Curie array", x=-71.55, y=-32.9, font="12p,Helvetica-Bold,red")


fig.show()
fig.savefig(figure_output_dir + '/map_of_earthquakes_Curie_GMT_0.png')


# %%
# try to download the C1.VA01 data from 2022-06-10 to 2022-06-14 (local time)