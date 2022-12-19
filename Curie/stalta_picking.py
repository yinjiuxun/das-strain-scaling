#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
from dateutil import parser
import obspy
import statsmodels.api as sm
import sys 
import h5py
import tqdm

from scipy.interpolate import interp1d

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.loading import save_rawevent_h5, load_event_data
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


event_folder = '/kuafu/EventData/Curie'  #'/kuafu/EventData/AlumRock5.1/MammothNorth'#'/kuafu/EventData/Ridgecrest' 
tt_output_dir = event_folder + '/theoretical_arrival_time0'

catalog = pd.read_csv(event_folder + '/catalog.csv')
das_info = pd.read_csv(event_folder + '/das_info.csv')
#%% plot event waveform
fig_folder = event_folder + '/event_examples'
mkdir(fig_folder)
tt_output_dir = event_folder + '/theoretical_arrival_time_calibrated'


ymin_list = [0, 0, 0]
for ii, i_event in enumerate([7]):#[1, 6, 7]
    event_now = catalog.iloc[i_event, :]
    event_id = event_now.event_id
    event_name = event_now.place
    magnitude = event_now.magnitude

    data, info = load_event_data(event_folder + '/data', event_id)
    DAS_channel_num = data.shape[1]
    dt = info['dt_s']
    das_time = np.arange(data.shape[0])*dt

    tt_1d = pd.read_csv(tt_output_dir + f'/1D_tt_{event_id}.csv')

    fig, gca = plt.subplots()
    plot_das_waveforms(data[::10, :], das_time[::10], gca, pclip=95)
    gca.plot(tt_1d.P_arrival, '--g')
    gca.plot(tt_1d.S_arrival, '-g')
    gca.set_ylim(ymin_list[ii], ymin_list[ii]+60)
    gca.set_title(f'M{magnitude}, {event_name}')




    plt.savefig(fig_folder + f'/{event_id}_waveforms.png', bbox_inches='tight')


# %%
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, carl_sta_trig, coincidence_trigger, plot_trigger

stalta_ratio = np.zeros(data.shape)

# Parameters of the STA/LTA
f_sample = 100

threshold_coincidence = 1
short_term = 0.1
long_term = 2
trigger_on = 6
trigger_off = 2

npt_extend = 500
for i in range(data.shape[1]):
    print(i)
    # test_data=np.zeros(shape=(data.shape[0]+npt_extend*2, ))
    # test_data[(npt_extend):(-npt_extend)] = data[:,i]

    test_data = data[:,i]
    stalta_ratio_temp = recursive_sta_lta(test_data, int(short_term * f_sample), int(long_term * f_sample))
    #stalta_ratio_temp = classic_sta_lta(test_data, int(short_term * f_sample), int(long_term * f_sample))
    # stalta_ratio[:, i] = stalta_ratio_temp[(npt_extend):(-npt_extend)]
    stalta_ratio[:, i] = stalta_ratio_temp

# cft = classic_sta_lta(st[0].data, int(short_term * f_sample), int(long_term * f_sample))
# cft = carl_sta_trig(st[0].data, int(short_term * 10), int(long_term * 10), 0.8, 0.8)
#%%
threshold = '_thresholded'
thresholod = 4

stalta_ratio_threshold = stalta_ratio.copy()
if threshold == '_thresholded':
    stalta_ratio_threshold[stalta_ratio_threshold<=thresholod]=0

stalta_ratio_peaks = np.max(stalta_ratio, axis=0)
ii_wrong_peak = stalta_ratio_peaks<=thresholod
stalta_ratio_peaks_index = np.argmax(stalta_ratio, axis=0)
peak_time = das_time[stalta_ratio_peaks_index[np.newaxis, :]]
peak_time[:, ii_wrong_peak] = np.nan

clb = plt.imshow(stalta_ratio_threshold, extent=[0, stalta_ratio.shape[1], 60, 0],aspect='auto', vmin=0, vmax=4)
plt.plot(tt_1d.P_arrival, '--g')
plt.plot(tt_1d.S_arrival, '-g')
# plt.plot(peak_time.T, 'r.')
plt.ylim(0, 30)
plt.plot(das_time, stalta_ratio[:, 2000], zorder=10)
plt.plot(das_time, data[:, 2000])
axins2 = inset_axes(plt.gca(),
                    width="2%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='lower right')
fig.colorbar(clb, cax=axins2, orientation="vertical", label='STA/LTA')
plt.savefig(fig_folder + f'/{event_id}_stalta_picking{threshold}.png', bbox_inches='tight')
# %%
