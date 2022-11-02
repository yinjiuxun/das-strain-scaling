#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import utm
import numpy as np
import h5py
import time
import tqdm
import glob
import obspy


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

#%% Functions to plot the events and DAS information
def plot_time_variation_of_events(catalog, figure_name):
    '''Time variation of events'''
    time_list = [obspy.UTCDateTime(time) for time in catalog[3]]
    time_span = np.array([time-time_list[0] for time in time_list])
    time_span_days = time_span/3600/24

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(time_span_days, catalog[7], 'o')
    ax.set_xlabel(f'Days from the {time_list[0]}')
    ax.set_ylabel('magnitude')
    plt.savefig(figure_name, bbox_inches='tight')


def plot_simple_map_of_das_and_events(DAS_index, DAS_lon, DAS_lat, catalog, figure_name):
    '''Simple map views of events'''
    fig, ax = plt.subplots(figsize=(7, 6))
    cmp = ax.scatter(DAS_lon, DAS_lat, s=10, c=DAS_index, cmap='jet')
    ax.scatter(catalog[5], catalog[4], s=10**(catalog[7]/5), c='k')
    fig.colorbar(cmp)
    plt.savefig(figure_name, bbox_inches='tight')

#%% Functions to plot the peak ampliutde variations
def plot_magnitude_distance_coverage(peak_amplitude_df, figure_name):
    '''Look at the data coverage in magnitude-distance space'''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(peak_amplitude_df.magnitude, peak_amplitude_df.distance_in_km, 
           s=3, c=peak_amplitude_df.event_label, marker='o', cmap='jet', alpha=0.1)

    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('log10(Distance) (log10(km))')

    plt.savefig(figure_name, bbox_inches='tight')


def plot_distance_variations(peak_amplitude_df, P_S_peak_keys, ymax, figure_name):
    '''Look at the rough variation of measured strain rate with distance'''
    fig, ax = plt.subplots(2, 1, figsize=(10, 20), sharex=True, sharey=True)
    ax[0].scatter(peak_amplitude_df.distance_in_km, peak_amplitude_df[P_S_peak_keys[0]], 
              s=10, c=peak_amplitude_df.event_label, marker='o', cmap='jet', alpha=0.1)
    ax[1].scatter(peak_amplitude_df.distance_in_km, peak_amplitude_df[P_S_peak_keys[1]], 
              s=20, c=peak_amplitude_df.event_label, marker='x', cmap='jet', alpha=0.1)

    ax[1].set_yscale('log')
    ax[1].set_xscale('log')

    pclip=99.99
    clipVal = np.percentile(peak_amplitude_df[P_S_peak_keys[1]], pclip)

    ax[0].set_ylabel('Maximum amplitude')
    ax[0].set_xlabel('log10(Distance)')
    ax[0].set_title('(a) Maximum ' + P_S_peak_keys[0])
    ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
    ax[1].set_title('(b) Maximum ' + P_S_peak_keys[1])
    ax[1].set_xlabel('log10(Distance)')
    ax[1].set_ylabel('Maximum amplitude')
    ax[1].set_ylim(top=ymax)

    plt.savefig(figure_name, bbox_inches='tight')

def plot_magnitude_variations(peak_amplitude_df, P_S_peak_keys, ymax, figure_name):
    ''' Look at the rough variation of measured strain rate with magnitude'''
    fig, ax = plt.subplots(2, 1, figsize=(10, 20), sharex=True, sharey=True)
    ax[0].scatter(peak_amplitude_df.magnitude, peak_amplitude_df[P_S_peak_keys[0]], 
              s=10, c=peak_amplitude_df.event_label, marker='o', cmap='jet', alpha=0.1)
    ax[1].scatter(peak_amplitude_df.magnitude, peak_amplitude_df[P_S_peak_keys[1]], 
              s=20, c=peak_amplitude_df.event_label, marker='x', cmap='jet', alpha=0.1)

    ax[1].set_yscale('log')

    pclip=99.99
    clipVal = np.percentile(peak_amplitude_df[P_S_peak_keys[1]], pclip)

    ax[0].set_ylabel('Maximum amplitude')
    ax[0].set_xlabel('Magnitude')
    ax[0].set_title('(a) Maximum ' + P_S_peak_keys[0])
    ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
    ax[1].set_title('(b) Maximum ' + P_S_peak_keys[1])
    ax[1].set_xlabel('Magnitude')
    ax[1].set_ylabel('Maximum amplitude')
    ax[1].set_ylim(top=ymax)

    plt.savefig(figure_name, bbox_inches='tight')

def plot_peak_time_distance_variations(peak_amplitude_df,ymax, figure_name):
    ''' Look at the rough variation of measured peak time with distance'''
    fig, ax = plt.subplots(2, 1, figsize=(10, 20), sharex=True, sharey=True)
    ax[0].scatter(peak_amplitude_df.distance_in_km, peak_amplitude_df.max_P_time, 
              s=10, c=peak_amplitude_df.event_label, marker='o', cmap='jet', alpha=0.1)
    ax[0].plot(peak_amplitude_df.distance_in_km, peak_amplitude_df.max_P_time2 + peak_amplitude_df.max_P_time, 'k.')
    
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

def plot_P_S_amplitude_ratio(peak_amplitude_df, P_S_peak_keys ,figure_name):
    '''Look at the distribution of P/S peak amplitude ratio'''
    P_S_ratio = peak_amplitude_df[P_S_peak_keys[0]] / peak_amplitude_df[P_S_peak_keys[1]]
    print(f'Median: {np.nanmedian(P_S_ratio)}')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(np.log10(P_S_ratio), range=(-2, 1), bins=50)
    ax.axvline(x=np.log10(np.nanmedian(P_S_ratio)), zorder=10, color='k')
    ax.set_xlabel('log10(P/S)')
    ax.set_ylabel('counts')

    plt.savefig(figure_name, bbox_inches='tight')


def plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict, y_S_predict, data_lim, figure_name, type='strain_rate'):
    if type == 'strain_rate':
        y_P = np.log10(peak_amplitude_df.peak_P)
        y_S = np.log10(peak_amplitude_df.peak_S)
    elif type == 'strain':
        y_P = np.log10(peak_amplitude_df.peak_P_strain)
        y_S = np.log10(peak_amplitude_df.peak_S_strain)
    else:
        raise NameError('type is wrong, use strain_rate or strain.')

    fig, ax = plt.subplots(2, 1, figsize=(10, 20), sharex=True, sharey=True)

    ax[0].plot([-10, 10], [-10, 10], '-k', zorder=1)
    ax[0].scatter(y_P, y_P_predict, s=10, c=peak_amplitude_df.event_label, marker='o', alpha=0.01, cmap='jet')
    ax[0].set_ylabel('P predicted log10(E)')
    ax[0].set_xlabel('P measured log10(E)')
    ax[0].xaxis.set_tick_params(which='both',labelbottom=True)

    ax[1].plot([-10, 10], [-10, 10], '-k', zorder=1)
    ax[1].scatter(y_S, y_S_predict, s=10, c=peak_amplitude_df.event_label, marker='o', alpha=0.01, cmap='jet')
    ax[1].set_ylabel('S predicted log10(E)')
    ax[1].set_xlabel('S measured log10(E)')

    ax[1].set_xlim(data_lim)
    ax[1].set_ylim(data_lim)

    plt.savefig(figure_name, bbox_inches='tight')