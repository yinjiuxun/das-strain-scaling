#%%
from operator import index
import pandas as pd
#from sep_util import read_file
import utm
import numpy as np
import glob

# fft
from scipy.fft import fft, fftfreq

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

#%%
from scipy.interpolate import interp1d

def extrapolate_arrival_time(ml_picks, kind='linear'):
    channel = np.array(ml_picks['station_name'])
    phase_index = np.array(ml_picks['phase_index'])
    channel_all = np.arange(0, DAS_channel_num)
    if kind is None:
        phase_index_extrap = np.ones(channel_all.shape) * -1
        phase_index_extrap[channel] = phase_index
        
    else:
        interp_f = interp1d(channel, phase_index, kind=kind, fill_value='extrapolate')
        phase_index_extrap = interp_f(channel_all).astype('int')
    
    return phase_index_extrap

# apply mask onto the data
def apply_mask_band(time_list, data_matrix, t1, t2, fill_value=np.nan):
    t1 = t1[:, np.newaxis]
    t2 = t2[:, np.newaxis]
    time_list = time_list[np.newaxis, :]
    
    # broadcast comparison
    mask_index = (time_list >= t1) & (time_list < t2)
    data_matrix_mask = data_matrix.copy()
    data_matrix_mask[~mask_index] = fill_value

    return data_matrix_mask

# apply mask onto the data
def apply_mask_after(time_list, data_matrix, t1, fill_value=np.nan):
    t1 = t1[:, np.newaxis]
    time_list = time_list[np.newaxis, :]
    
    # broadcast comparison
    mask_index = time_list >= t1
    data_matrix_mask = data_matrix.copy()
    data_matrix_mask[mask_index] = fill_value

    return data_matrix_mask


def extract_maximum_amplitude(time_list, data_matrix, t1, t2):
    data_matrix_mask = apply_mask_band(time_list, data_matrix, t1, t2)
    max_amplitude = np.nanmax(data_matrix_mask, axis=1)
    return max_amplitude

#%%
catalog_file = '/home/yinjx/notebooks/strain_scaling/Ridgecrest_das_catalog_M2_M8.txt'
catalog = pd.read_csv(catalog_file, sep='\s+', header=None, skipfooter=1, engine='python')
catalog_select = catalog[(catalog[7] >= 3) & (catalog[6] > 1)] # choose the event with magnitude > 3.5

eq_num = catalog_select.shape[0]
eq_id = np.array(catalog_select[0])
eq_time = np.array(catalog_select[3])
eq_lat = np.array(catalog_select[4])
eq_lon = np.array(catalog_select[5])
eq_dep = np.array(catalog_select[6])
eq_mag = np.array(catalog_select[7])

print(f'Total number of events: {eq_num}')

#%%
# load the DAS channel location
DAS_info = np.genfromtxt('/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Ridgecrest_ODH3.txt')

DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info[:, 0].astype('int')
DAS_lon = DAS_info[:, 1]
DAS_lat = DAS_info[:, 2]

#%% 
ii_eq = 32
eq_id_temp = [eq_id[ii_eq]]
extrapolate_ml_picking = False

# path of the ML picking 
ml_pick_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/picking_Weiqiang/picks_yinjx_all'

# Check the segmented 50Hz data
das_path = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/ML_picking_data_eye_picked'

das_dt = 0.02 # downsampled to 50 Hz

# mean spectrum
peak_freq_mean = []
magnitude_mean = []
# individual channel
peak_freq_all = np.array([])
magnitude_all = np.array([])

time_from_P = 0.5

for ii_eq, eq_id_current in enumerate(eq_id):
      
    try: 
        # look for the corresponding ML picks
        ml_picks_file = glob.glob(ml_pick_dir + f'/*{eq_id_current}.csv')
        ml_picks = pd.read_csv(ml_picks_file[0])

        # extract the picked information
        ml_picks_p = ml_picks[ml_picks['phase_type'] == 'p']
        ml_picks_s = ml_picks[ml_picks['phase_type'] == 's']

        # remove some duplicated picks, only keep those with higher probability
        ml_picks_p = ml_picks_p.drop_duplicates(subset=['station_name'], keep='first')
        ml_picks_s = ml_picks_s.drop_duplicates(subset=['station_name'], keep='first')
        
        if extrapolate_ml_picking:
            # extrapolate the index in case the ML picking is imcomplete
            event_arrival_P_ml_index = extrapolate_arrival_time(ml_picks_p, kind='nearest')
            event_arrival_S_ml_index = extrapolate_arrival_time(ml_picks_s, kind='nearest')
            channel_of_picks_P = np.arange(DAS_channel_num)
            channel_of_picks_S = np.arange(DAS_channel_num)
            fig_name = f'{eq_id_current}_ml_extrap_nearest.png'
        else:
            # not extrapolate the index in case the ML picking is imcomplete
            event_arrival_P_ml_index = extrapolate_arrival_time(ml_picks_p, kind=None)
            event_arrival_S_ml_index = extrapolate_arrival_time(ml_picks_s, kind=None)
            channel_of_picks_P = ml_picks_p['station_name']
            channel_of_picks_S = ml_picks_s['station_name']
            fig_name = f'{eq_id_current}_ml.png'
    
    except:
        print(f'ML picking {eq_id_current} not found, skip...')
        continue
        
    try:      
        # look for the corresponding data file
        file_name = glob.glob(das_path + f'/*{eq_id_current}.npz')
        temp = np.load(file_name[0])
        data_diff = temp[temp.files[0]]
        
        # time information 
        das_time = np.arange(data_diff.shape[0]) * das_dt

        # get the ML pick time
        event_arrival_P_ml = das_time[event_arrival_P_ml_index.astype('int')]
        event_arrival_S_ml = das_time[event_arrival_S_ml_index.astype('int')]

        # Slight modify only for plotting purpose
        event_arrival_P_ml[event_arrival_P_ml == das_time[-1]] = np.nan
        event_arrival_S_ml[event_arrival_S_ml == das_time[-1]] = np.nan
        
        print(f'Event {eq_id_current} done!')
        
    except:
        print(f'DAS file {eq_id_current} not found, skip...')
        continue

    masked_P = apply_mask_band(das_time, data_diff.T, event_arrival_P_ml-1, event_arrival_P_ml+time_from_P, fill_value=0)  
    masked_P_fft = np.abs(fft(masked_P, axis=1))
    fft_freq = fftfreq(len(das_time), das_dt)

    # Only keep the positive frequency
    masked_P_fft = masked_P_fft[:,fft_freq>0]
    fft_freq = fft_freq[fft_freq>0]
    
    

    peak_ii = np.argmax(masked_P_fft, axis=1)
    kept_channel = peak_ii>0
    peak_ii = peak_ii[kept_channel]


    peak_freq_all = np.concatenate((peak_freq_all, fft_freq[peak_ii]), axis=0)
    magnitude_all= np.concatenate((magnitude_all, eq_mag[ii_eq] * np.ones(peak_ii.shape)), axis=0)

    mean_spectrum = np.nanmean(masked_P_fft[kept_channel, :], axis=0)
    peak_ii_mean = np.argmax(mean_spectrum)
    peak_freq_mean.append(fft_freq[peak_ii_mean])
    magnitude_mean.append(eq_mag[ii_eq])

# flatten the results
peak_freq_all_flatten = peak_freq_all
magnitude_all_flatten = magnitude_all
fig, ax = plt.subplots(figsize=(8 , 8))
ax.plot(0, np.nan, 'bx', label='individual channel')
ax.semilogy(magnitude_all_flatten, peak_freq_all_flatten,'bx', alpha=0.01)
ax.semilogy(magnitude_mean, peak_freq_mean,'ro', alpha=1, label='averaged spectrum')
ax.set_ylim(ymin=1)
ax.set_ylabel('predominant frequency')
ax.set_xlabel('magnitude')
ax.grid()
ax.legend()
ax.set_title(f'Predominat frequency {time_from_P} s after P arrival')
plt.savefig(das_path + f'/predominant_frequency_{time_from_P}s_after_P.png', bbox_inches='tight')


# %% Investigate a specified event
time_from_P = 2
eq_id_current = 38669279
print(np.where(eq_id==eq_id_current))
index_eq = np.where(eq_id==eq_id_current)[0][0]

#%%
file_name = glob.glob(das_path + f'/*{eq_id_current}.npz')
temp = np.load(file_name[0])
data_diff = temp[temp.files[0]]

# time information 
das_time = np.arange(data_diff.shape[0]) * das_dt

# get the ML pick time
ml_picks_file = glob.glob(ml_pick_dir + f'/*{eq_id_current}.csv')
ml_picks = pd.read_csv(ml_picks_file[0])

# extract the picked information
ml_picks_p = ml_picks[ml_picks['phase_type'] == 'p']
ml_picks_s = ml_picks[ml_picks['phase_type'] == 's']

# remove some duplicated picks, only keep those with higher probability
ml_picks_p = ml_picks_p.drop_duplicates(subset=['station_name'], keep='first')
ml_picks_s = ml_picks_s.drop_duplicates(subset=['station_name'], keep='first')
event_arrival_P_ml_index = extrapolate_arrival_time(ml_picks_p, kind=None)
event_arrival_S_ml_index = extrapolate_arrival_time(ml_picks_s, kind=None)

event_arrival_P_ml = das_time[event_arrival_P_ml_index.astype('int')]
event_arrival_S_ml = das_time[event_arrival_S_ml_index.astype('int')]

# Slight modify only for plotting purpose
event_arrival_P_ml[event_arrival_P_ml == das_time[-1]] = np.nan
event_arrival_S_ml[event_arrival_S_ml == das_time[-1]] = np.nan

masked_P = apply_mask_band(das_time, data_diff.T, event_arrival_P_ml-1, event_arrival_P_ml+time_from_P, fill_value=0)

# Show data
fig, ax1 = plt.subplots(2, 1, figsize=(16,16))
pclip=99.5
clipVal = np.percentile(np.absolute(data_diff), pclip)
# Vx
ax1[0].imshow(data_diff, 
            extent=[0, data_diff.shape[1], das_time[-1], das_time[0]],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

ax1[0].plot(event_arrival_P_ml, '--k', linewidth=2, zorder=10)
ax1[0].plot(event_arrival_S_ml, '-k', linewidth=2, zorder=10)
ax1[0].set_title(f'Event ID: {eq_id_current}')
ax1[0].set_ylabel("Time [s]")
ax1[0].grid()

ax1[1].imshow(masked_P.T, 
            extent=[0, data_diff.shape[1], das_time[-1], das_time[0]],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

ax1[1].plot(event_arrival_P_ml, '--k', linewidth=2, zorder=10)
ax1[1].plot(event_arrival_S_ml, '-k', linewidth=2, zorder=10)

ax1[1].set_xlabel("Channel number")
ax1[1].set_ylabel("Time [s]")
ax1[1].grid()
plt.savefig(das_path + f'/data_example_{eq_id_current}_{time_from_P}s_after_P.png', bbox_inches='tight')
# %% Plot spectrum

masked_P_fft = np.abs(fft(masked_P, axis=1))
fft_freq = fftfreq(len(das_time), das_dt)

# Only keep the positive frequency
masked_P_fft = masked_P_fft[:,fft_freq>0]
fft_freq = fft_freq[fft_freq>0]
mean_spectrum = np.nanmean(masked_P_fft, axis=0)

fig2, ax = plt.subplots(figsize=(8,8))
cmp = plt.cm.jet(np.arange(0, 1, DAS_channel_num))


ax.loglog(fft_freq, masked_P_fft[::10,:].T, '-', linewidth=0.5, alpha=0.5)
ax.loglog(fft_freq, mean_spectrum, '-k')
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('spectrum')
ax.set_title(f'Spectrum {time_from_P} s after P arrival')
ax.set_xlim(xmin=1)
plt.savefig(das_path + f'/data_example_{eq_id_current}_{time_from_P}s_after_P_spectrum.png', bbox_inches='tight')
# %%
