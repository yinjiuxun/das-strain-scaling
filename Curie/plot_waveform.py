
#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser
import obspy
import statsmodels.api as sm
import sys 

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.loading import load_event_data, load_phasenet_pick
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


#%%
#load event waveform to plot
event_folder = '/kuafu/EventData/Curie'  #'/kuafu/EventData/AlumRock5.1/MammothNorth'#'/kuafu/EventData/Ridgecrest' 
tt_dir = event_folder +  '/theoretical_arrival_time0' 
catalog = pd.read_csv(event_folder + '/catalog.csv')
DAS_info = pd.read_csv('/kuafu/EventData/Curie/das_info.csv')
das_waveform_path = event_folder + '/data_raw'


DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['index'][::2]
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

# #/kuafu/EventData/AlumRock5.1/MammothNorth
# test_event_id, given_range_P, given_range_S, ymin, ymax = 73799091, None, None, 0, 90 
# tpshift, tsshift = -2, -5
# test_event_id_list, given_range_P_list, given_range_S_list, ymin_list, ymax_list = append_list(test_event_id, given_range_P, given_range_S, ymin, ymax)

# Curie events
test_event_id_list = list(catalog.event_id)
for ii in range(len(test_event_id_list)):
    given_range_P_list.append(None)
    given_range_S_list.append(None)
    if ii == 1:
        ymin_list.append(5)
        ymax_list.append(30)
    elif ii == 6:
        ymin_list.append(0)
        ymax_list.append(25)
    elif ii == 7:
        ymin_list.append(15)
        ymax_list.append(60)
    else:
        ymin_list.append(0)
        ymax_list.append(60)
#%%
# plot waveform
time_drift = {'9000':12.5, '9001':12.5, '9002':np.nan, '9003':np.nan, '9004':9, '9005':8.5, '9006':9, '9007':9}


# load the travel time and process
def remove_ml_tt_outliers(ML_picking, das_dt, tdiff=10, given_range=None):
    temp = ML_picking.drop(index=ML_picking[abs(ML_picking.phase_index - ML_picking.phase_index.median())*das_dt >= tdiff].index)
    if given_range:
        try:
            temp = temp[(temp.phase_index>=given_range[0]/das_dt) & (temp.phase_index<=given_range[1]/das_dt)]
        except:
            print('cannot specify range, skip...')
    return temp


for i_event in [1, 6, 7]:#range(len(test_event_id_list)):
    test_event_id = test_event_id_list[i_event]
    print(test_event_id)
    given_range_P = given_range_P_list[i_event]
    given_range_S = given_range_S_list[i_event]
    ymin, ymax = ymin_list[i_event], ymax_list[i_event]


    event_info = catalog[catalog.event_id == test_event_id]
    try:
        strain_rate, info = load_event_data(das_waveform_path, test_event_id)
        das_dt = info['dt_s']
        nt = strain_rate.shape[0]
        das_time = np.arange(nt) * das_dt-30


        # plot some waveforms and picking
        fig, gca = plt.subplots(figsize=(10, 4))
        title_text = f'M{event_info.iloc[0, :].magnitude}, {event_info.iloc[0, :].place}'
        plot_das_waveforms(strain_rate, das_time, gca, title=title_text, pclip=95, ymin=ymin, ymax=ymax)
    except:
        print(f'Event {test_event_id} data not found, skip...')
        continue

    # load the ML phase picking
    try:
        ML_picking_dir = event_folder + '/picks_phasenet_das'
        tt_tp = np.zeros(shape=DAS_channel_num)*np.nan
        tt_ts = tt_tp.copy()

        ML_picking = pd.read_csv(ML_picking_dir + f'/{test_event_id}.csv')
        ML_picking = ML_picking[ML_picking.channel_index < DAS_channel_num]

        ML_picking_P = ML_picking[ML_picking.phase_type == 'P']
        ML_picking_S = ML_picking[ML_picking.phase_type == 'S']
        ML_picking_P = remove_ml_tt_outliers(ML_picking_P, das_dt, tdiff=25, given_range=given_range_P)
        ML_picking_S = remove_ml_tt_outliers(ML_picking_S, das_dt, tdiff=25, given_range=given_range_S)

        tt_tp[ML_picking_P.channel_index] = das_time[ML_picking_P.phase_index]
        tt_ts[ML_picking_S.channel_index] = das_time[ML_picking_S.phase_index]
        gca.plot(tt_tp, '--k', linewidth=3, label='ML-picked P')
        gca.plot(tt_ts, '-k', linewidth=3, label='ML-picked S')
    except:
        print('Cannot find the ML travel time, skip...')

    # also load the theoretical time
    try:
        cvm_tt = pd.read_csv(tt_dir + f'/1D_tt_{test_event_id}.csv')
        tt_tp_vm = np.array(cvm_tt.P_arrival)
        tt_ts_vm = np.array(cvm_tt.S_arrival)
        gca.plot(tt_tp_vm, '--g', linewidth=3, label=f"theoretical P")
        gca.plot(tt_ts_vm, '-g', linewidth=3, label=f"theoretical S")
    except:
        print('Cannot find the theoretical travel time, skip...')

    gca.legend(loc=3, fontsize=10)
    gca.invert_yaxis()
    mkdir(event_folder + '/event_examples')
    plt.savefig(event_folder + f'/event_examples/{test_event_id}_ML_paper.png', bbox_inches='tight')





# # shift the ML picking
# ML_picking_shift = {'9001':34.5, '9006':4, '9007': 40.5}
# for i_event in [1, 6, 7]:#range(len(test_event_id_list)):
#     test_event_id = test_event_id_list[i_event]
#     ML_picking_dir1 = event_folder + '/picks_phasenet_das_safe_copy'
#     ML_picking_dir2 = event_folder + '/picks_phasenet_das'
#     ML_picking = pd.read_csv(ML_picking_dir1 + f'/{test_event_id}.csv')
#     ML_picking['phase_index'] = ML_picking['phase_index'] + int(ML_picking_shift[str(test_event_id)]/das_dt)

#     ML_picking.to_csv(ML_picking_dir2 + f'/{test_event_id}.csv')
# %%
