#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser
import obspy
import statsmodels.api as sm
import sys 

sys.path.append("../")
from utility.general import mkdir
from utility.loading import load_event_data
from utility.processing import remove_outliers, filter_event
from utility.plotting import plot_das_waveforms

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
#load event waveform to plot
event_folder = '/kuafu/EventData/LA_Google' 
tt_dir = event_folder +  '/theoretical_arrival_time' 
catalog = pd.read_csv(event_folder + '/catalog.csv')
das_waveform_path = event_folder + '/data'

DAS_info = pd.read_csv('/kuafu/EventData/LA_Google/das_info.csv')
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info.index
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

#%%
# work out list to plot
test_event_id_list = []
given_range_P_list, given_range_S_list = [], []
ymin_list, ymax_list = [], []

def append_list(test_event_id, given_range_P, given_range_S, ymin, ymax):
    test_event_id_list.append(test_event_id)
    given_range_P_list.append(given_range_P)
    given_range_S_list.append(given_range_S)
    ymin_list.append(ymin)
    ymax_list.append(ymax)
    return test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list

given_range_P=None
# # Load the DAS data
# test_event_id, given_range_S, ymin, ymax = 40241984, None, 50, 70 #40241984(2.88 from offshore) 
# test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)

# Offshore events
test_event_id, given_range_P, given_range_S, ymin, ymax = 39970455, None, None, 55, 70 #39970455 (M2.79 from SW offshore)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 40296896, None, [0, 60], 0, 90 #40182560(2.36 from offshore) 
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 40241984, None, [50, 70], 0, 90 #40241984(2.88 from offshore) 
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)

# # The following 4 came from almost the same location
test_event_id, given_range_P, given_range_S, ymin, ymax = 40194736, None, [55, 65], 40, 90 #40194736 (3.98)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 40182560, None, [55, 60], 40, 90 #40182560(3.86)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 40194848, None, [55, 59], 40, 90 #40194848(3.05)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 40287104, None, [55, 60], 40, 90 #40287104(2.84)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)

# # one from the N
test_event_id, given_range_P, given_range_S, ymin, ymax = 39974975, None, [40, 46], 35, 55 # (One M2.6 from N)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)

# # Another one from NW
test_event_id, given_range_P, given_range_S, ymin, ymax = 39914199, None, None, 40, 80 # (Another one M3 from NW)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)

# # The following are close events with P
test_event_id, given_range_P, given_range_S, ymin, ymax = 39983383, None, None, 32, 46 # (2.32, below the array)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 40180104, None, [30, 40], 30, 50 # (2.59, Hollywood)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 39929895, None, None, 30, 50 # (3.26, South Gate)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 39974135, None, [55, 80], 40, 80 # (3.28, Ontario)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)
test_event_id, given_range_P, given_range_S, ymin, ymax = 39988031, None, [32, 38], 30, 45 # (2.09, View Park-Windsor Hills)
test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)

#%%
# plot waveform
# load the travel time and process
def remove_ml_tt_outliers(ML_picking, das_dt, tdiff=10, given_range=None):
    temp = ML_picking.drop(index=ML_picking[abs(ML_picking.phase_index - ML_picking.phase_index.median())*das_dt >= tdiff].index)
    if given_range:
        try:
            temp = temp[(temp.phase_index>=given_range[0]/das_dt) & (temp.phase_index<=given_range[1]/das_dt)]
        except:
            print('cannot specify range, skip...')
    return temp


for i_event in range(len(test_event_id_list)):
    print(test_event_id)
    test_event_id = test_event_id_list[i_event]
    given_range_P = given_range_P_list[i_event]
    given_range_S = given_range_S_list[i_event]
    ymin, ymax = ymin_list[i_event], ymax_list[i_event]


    event_info = catalog[catalog.event_id == test_event_id]
    strain_rate, info = load_event_data(das_waveform_path, test_event_id)
    strain_rate = strain_rate[:, DAS_index]
    das_dt = info['dt_s']
    nt = strain_rate.shape[0]
    das_time = np.arange(nt) * das_dt

    # plot some waveforms and picking
    ML_picking_dir = event_folder + '/picks_phasenet_das'
    tt_tp = np.zeros(shape=DAS_channel_num)*np.nan
    tt_ts = tt_tp.copy()

    ML_picking = pd.read_csv(ML_picking_dir + f'/{test_event_id}.csv')
    ML_picking_P = ML_picking[ML_picking.phase_type == 'P']
    ML_picking_S = ML_picking[ML_picking.phase_type == 'S']
    ML_picking_P = remove_ml_tt_outliers(ML_picking_P, das_dt, tdiff=25, given_range=given_range_P)
    ML_picking_S = remove_ml_tt_outliers(ML_picking_S, das_dt, tdiff=25, given_range=given_range_S)

    tt_tp[ML_picking_P.channel_index] = das_time[ML_picking_P.phase_index]
    tt_ts[ML_picking_S.channel_index] = das_time[ML_picking_S.phase_index]

    fig, gca = plt.subplots(figsize=(10, 6))
    plot_das_waveforms(strain_rate, das_time, gca, title=f'{test_event_id}, M{event_info.iloc[0, :].magnitude}', pclip=95, ymin=ymin, ymax=ymax)
    gca.plot(tt_tp, '--k', linewidth=2)
    gca.plot(tt_ts, '-k', linewidth=2)
    gca.invert_yaxis()

    mkdir(event_folder + '/event_examples')
    plt.savefig(event_folder + f'/event_examples/{test_event_id}.png', bbox_inches='tight')











# #%%
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
