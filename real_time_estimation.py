#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

from obspy.geodetics import locations2degrees

# import the plotting functions
from plotting_functions import *
# import the utility functions
from utility_functions import *

#%% Define functions
# Use the predicted strain to calculate magnitude
def estimate_magnitude_from_data(peak_amplitude_df, reg, type, fitting_type='without_site', site_term_column='region_site'):
    if fitting_type == 'with_site':
        # get the annoying categorical keys
        try:
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
        except:
            raise NameError(f"Index {site_term_column} doesn't exist!")

        if type == 'P':    
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')

    elif fitting_type == 'with_attenuation':
        # get the annoying categorical keys C(region)[ridgecrest]:distance_in_km 
        try:
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
            site_attenu_keys = np.array([f'C(region)[{region}]:distance_in_km' for region in peak_amplitude_df['region']])
        except:
            raise NameError(f"Index {site_term_column} doesn't exist!")

        if type == 'P':    
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km) \
                        - np.array(reg.params[site_attenu_keys]) * np.array(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - np.array(reg.params[site_term_keys]) \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km) \
                        - np.array(reg.params[site_attenu_keys]) * np.array(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')
        
    elif fitting_type == 'without_site':
        if type == 'P':
            M_predict = (np.log10(peak_amplitude_df.peak_P) \
                        - reg.params['Intercept'] \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        elif type == 'S':
            M_predict = (np.log10(peak_amplitude_df.peak_S) \
                        - reg.params['Intercept'] \
                        - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                        / reg.params['magnitude']
        else:
            raise NameError(f'{type} is not defined! Only "P" or "S"')

    else:
        raise NameError('Fitting type is undefined!')
        
    return M_predict

def get_mean_magnitude(peak_amplitude_df, M_predict):
    temp_df = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df['predicted_M'] = M_predict
    temp_df = temp_df.groupby(temp_df['event_id']).aggregate('mean')

    temp_df2 = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df2['predicted_M_std'] = M_predict
    temp_df2 = temp_df2.groupby(temp_df2['event_id']).aggregate(np.std)

    temp_df = pd.concat([temp_df, temp_df2['predicted_M_std']], axis=1)
    return temp_df

# ========================== work on Combined results from both Ridgecrest and Olancha ================================
# First check how well the regression relation can be used to calculate Magnitude
# Then try to use the regression relation from small events to predict the larger ones
#%% load the results from combined regional site terms t
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate_refined'#'/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RO'
regression_dir = 'regression_results_smf_M4'
site_term_column = 'region_site'
fitting_type = 'with_site'

catalog_file = '/home/yinjx/notebooks/strain_scaling/Ridgecrest_das_catalog_M2_M8.txt'
catalog = pd.read_csv(catalog_file, sep='\s+', header=None, skipfooter=1, engine='python')

# load the DAS array information
DAS_info = np.genfromtxt('/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Ridgecrest_ODH3.txt')

DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info[:, 0].astype('int')
DAS_lon = DAS_info[:, 1]
DAS_lat = DAS_info[:, 2]

das_path = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/ML_picking_data_eye_picked'
ml_pick_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/picking_Weiqiang/picks_yinjx_all'

#%% load the results from combined regional site terms t
results_output_dir = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling'#'/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RO'
regression_dir = 'regression_results_smf_M4'
site_term_column = 'region_site'
fitting_type = 'with_site'

catalog_file = '/home/yinjx/notebooks/strain_scaling/Olancha_das_catalog_M2_M8.txt'
catalog = pd.read_csv(catalog_file, sep='\s+', header=None, skipfooter=1, engine='python')
catalog = catalog.dropna()

# load the DAS array information
DAS_info_old = pd.read_csv('/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Olancha_Plexus.txt', 
                            sep='\s+', header=None, engine='python')

DAS_channel_num_old = DAS_info_old.shape[0]
DAS_index_old = DAS_info_old.iloc[:, 0].astype('int')
DAS_lon_old = DAS_info_old.iloc[:, 1]
DAS_lat_old = DAS_info_old.iloc[:, 2]

# new generation
DAS_info_new = pd.read_csv('/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Olancha_Plexus_100km.txt',
                            sep=',', engine='python')
DAS_info_new = DAS_info_new.dropna()
DAS_info_new = DAS_info_new[(DAS_info_new.status == 'good') & (DAS_info_new.channel < 5000)]

DAS_channel_num_new = DAS_info_new.shape[0]
DAS_index_new = DAS_info_new.iloc[:, 0].astype('int')
DAS_lon_new = DAS_info_new.iloc[:, 3]
DAS_lat_new = DAS_info_new.iloc[:, 2]

das_path = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling/ML_picking_data_M_3+'
ml_pick_dir = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling/picks_yinjx_Olancha_Plexus_100km'

#%% make figure output directory
fig_output_dir = results_output_dir + '/' + regression_dir + '/estimated_M'
if not os.path.exists(fig_output_dir):
    os.mkdir(fig_output_dir)

#%% ========================== Real time estimation ================================
# load event information 
nearby_channel_number = 10
peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
event_peak_df = peak_amplitude_df[peak_amplitude_df.magnitude > 4]
event_id_list = event_peak_df.event_id.unique().astype('int')

#%%
for test_event_id in [39464303]:#event_id_list:
    # try:
    #test_event_id = 39493944 # 38548295 # 39462536 #39493944 # [39462536]
    eq_info = catalog[catalog[0] == test_event_id]
    eq_lat = eq_info[4] # lat
    eq_lon = eq_info[5] # lon
    eq_mag = eq_info[7].astype('float') # catalog magnitude
    eq_time = eq_info[3].values[0] # eq time

    # load regression parameters
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")

    # Load the DAS data
    # Check the segmented 50Hz data
    
    file_name = glob.glob(das_path + f'/*{test_event_id}.npz')
    temp = np.load(file_name[0])
    data_diff = temp[temp.files[0]]
    das_dt = 0.02
    das_time0 = np.arange(data_diff.shape[0]) * das_dt

    # For Olancha only
    if data_diff.shape[1] == 4772:
        DAS_lat = DAS_lat_new
        DAS_lon = DAS_lon_new
        DAS_index = DAS_index_new
        to_nano_factor = 1e3
    else:
        DAS_lat = DAS_lat_old
        DAS_lon = DAS_lon_old
        DAS_index = DAS_index_old
        to_nano_factor = 1e9

    data_diff = data_diff * to_nano_factor

    # get distance from earthquake to each channel
    distance_to_source = locations2degrees(DAS_lat, DAS_lon, eq_lat, eq_lon) * 113 # in km
    distance_to_source = distance_to_source[np.newaxis, :]

    # TODO: HERE need to rethink, maybe the sliding time windows with overlap help??
    # convert data matrix to peak amplitude matrix
    # data_peak_mat = np.zeros(data_diff.shape)
    # for ii_win in range(data_diff.shape[0]):
    #     data_peak_mat[ii_win, :] = np.max(abs(data_diff[:(ii_win+1), :]), axis=0)

    time_step = 200
    data_peak_mat = np.zeros((np.ceil(data_diff.shape[0]/time_step).astype('int'), data_diff.shape[1]))
    for i, ii_win in enumerate(range(0, data_diff.shape[0], time_step)):
        data_peak_mat[i, :] = np.max(abs(data_diff[ii_win:(ii_win+time_step), :]), axis=0)    
    das_time = das_time0[::time_step]

    

    ml_picks_file = glob.glob(ml_pick_dir + f'/*{test_event_id}.csv')
    ml_picks = pd.read_csv(ml_picks_file[0])

    # extract the picked information
    ml_picks_p = ml_picks[ml_picks['phase_type'] == 'p']
    ml_picks_s = ml_picks[ml_picks['phase_type'] == 's']
    P_arrival_index = np.median(ml_picks_p.phase_index).astype('int')
    S_arrival_index = np.median(ml_picks_s.phase_index).astype('int')

    P_arrival_approx = das_time0[P_arrival_index]
    S_arrival_approx = das_time0[S_arrival_index]

    # get the site terms
    if nearby_channel_number == -1: # when nearby_channel_number == -1, combined all channels!
        nearby_channel_number = DAS_index.max()+1
        temp1= np.arange(0, DAS_index.max())
    else:
        nearby_channel_number = DAS_index.max()+1
        temp1= np.arange(0, DAS_index.max()+1) # original channel number
    temp2 = temp1 // nearby_channel_number # combined channel number
    site_term_keys = np.array([f'C(combined_channel_id)[{site_term}]' for site_term in temp2])

    site_terms = regP.params[site_term_keys]
    site_terms = site_terms[DAS_index]
    site_terms = site_terms[np.newaxis, :]

    mag_estimate = (np.log10(data_peak_mat+1e-12) - site_terms - np.log10(distance_to_source)*regP.params['np.log10(distance_in_km)'])/regP.params['magnitude']

    #mag_estimate = (np.log10(data_peak_mat+1e-12) - regP.params['C(region_site)[ridgecrest-0]'] - np.log10(distance_to_source)*regP.params['np.log10(distance_in_km)'])/regP.params['magnitude']
    median_mag = np.median(mag_estimate, axis=1)
    mean_mag = np.mean(mag_estimate, axis=1)
    #%%
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(das_time, mag_estimate, '-k', linewidth=0.1, alpha=0.1)
    ax.plot(das_time, median_mag, '--r', linewidth=2, alpha=0.5, zorder=3)
    ax.plot(das_time, mean_mag, '-r', linewidth=2, alpha=0.5, zorder=3)
    ax.vlines(x=[P_arrival_approx, S_arrival_approx], ymax=10, ymin=0)
    ax.hlines(y=[eq_mag], xmin=-10, xmax=100)
    ax.set_ylim(0, eq_mag.values*1.4)
    ax.set_xlim(-10, das_time[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Estimated Magnitude')
    ax.set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}')

    # plt.savefig(fig_output_dir + f'/{test_event_id}_estimated_mag.png')
    # plt.close('all')

    print(f'{test_event_id} done!')
    # except:
    #     continue



















# %% A quick look at continous data

def read_PASSCAL_segy(infile, nTraces, nSample, TraceOff=0):
    """Function to read PASSCAL segy raw data"""
    data = np.zeros((nTraces, nSample), dtype=np.float32)
    gzFile = False
    if infile.split(".")[-1] == "segy":
        fid = open(infile, 'rb')
    elif infile.split(".")[-1] == "gz":
        gzFile = True
        fid = gzip.open(infile, 'rb')
    fid.seek(3600)
    # Skipping traces if necessary
    fid.seek(TraceOff*(240+nSample*4),1)
    # Looping over traces
    for ii in range(nTraces):
        fid.seek(240, 1)
        if gzFile:
            # np.fromfile does not work on gzip file
            BinDataBuffer = fid.read(nSample*4) # read binary bytes from file
            data[ii, :] = struct.unpack_from(">"+('f')*nSample, BinDataBuffer)
        else:
            data[ii, :] = np.fromfile(fid, dtype=np.float32, count=nSample)
    fid.close()
    return data

    
import glob
files = glob.glob('/kuafu/zshen/Ridgecrest_data/1hoursegy/total/*.segy')
files.sort()
print(f'Total file number: {len(files)}')
print(files[0])
print(files[-1])

#%%
DAS_info = np.genfromtxt('/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Ridgecrest_ODH3.txt')

DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info[:, 0].astype('int')
DAS_lon = DAS_info[:, 1]
DAS_lat = DAS_info[:, 2]

#%% Convert the event time to obspy UTCDateTime and also find the corresponding DAS file name
import datetime
import obspy

peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
event_peak_df = peak_amplitude_df[peak_amplitude_df.magnitude > 4]
event_id_list = event_peak_df.event_id.unique().astype('int')


# load event information 
catalog_file = '/home/yinjx/notebooks/strain_scaling/Ridgecrest_das_catalog_M2_M8.txt'
catalog = pd.read_csv(catalog_file, sep='\s+', header=None, skipfooter=1, engine='python')

for test_event_id in event_id_list:
    #test_event_id = 38683914
    print(test_event_id)

    #test_event_id = 39493944 # 38548295 # 39462536 #39493944 # [39462536]
    eq_info = catalog[catalog[0] == test_event_id]
    eq_lat = eq_info[4] # lat
    eq_lon = eq_info[5] # lon
    eq_mag = eq_info[7] # catalog magnitude
    eq_time = eq_info[3].values[0] # eq time

    # get distance from earthquake to each channel
    distance_to_source = locations2degrees(DAS_lat, DAS_lon, eq_lat, eq_lon) * 113 # in km
    distance_to_source = distance_to_source[np.newaxis, :]


    # event time in obspy.UTCDateTime
    eq_das_files = datetime.datetime.strptime(eq_time[:-4],'%Y/%m/%d,%H:%M:%S').strftime('%Y%m%d%H.segy')
    #%%
    das_path = '/kuafu/zshen/Ridgecrest_data/1hoursegy/total/'

    data0 = read_PASSCAL_segy(das_path + eq_das_files, 1250, 900000, 0)
    data0 = data0[DAS_index, :]

    das_dt = 3600 / data0.shape[1]
    data_diff = np.diff(data0, axis=1)/das_dt

    # time information 
    das_time = np.linspace(das_dt,3600-das_dt,data_diff.shape[1])

    # Downsample from 250 Hz to 50 Hz
    data_diff = data_diff[:, ::5].T
    das_time = das_time[::5]

    # convert data matrix to peak amplitude matrix
    time_step = 10
    data_peak_mat = np.zeros((data_diff.shape[0]//time_step, data_diff.shape[1]))
    for i, ii_win in enumerate(range(0, data_diff.shape[0], time_step)):
        data_peak_mat[i, :] = np.max(abs(data_diff[ii_win:(ii_win+time_step), :]), axis=0)    

    das_time = das_time[::time_step]

    mag_estimate = (np.log10(data_peak_mat+1e-12) - regP.params['C(combined_channel_id)[0]'] - np.log10(distance_to_source)*regP.params['np.log10(distance_in_km)'])/regP.params['magnitude']

    #mag_estimate = (np.log10(data_peak_mat+1e-12) - regP.params['C(region_site)[ridgecrest-0]'] - np.log10(distance_to_source)*regP.params['np.log10(distance_in_km)'])/regP.params['magnitude']
    median_mag = np.median(mag_estimate, axis=1)
    mean_mag = np.mean(mag_estimate, axis=1)
    #%%
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(das_time, mag_estimate[:, ::10], '-k', linewidth=0.1, alpha=0.01)
    ax.plot(das_time, median_mag, '--r', linewidth=2, alpha=0.5, zorder=3)
    ax.plot(das_time, mean_mag, '-r', linewidth=2, alpha=0.5, zorder=3)
    # ax.vlines(x=[P_arrival_approx, S_arrival_approx], ymax=10, ymin=0)
    ax.hlines(y=[eq_mag], xmin=-10, xmax=das_time[-1])
    # ax.set_ylim(0, eq_mag.values*1.4)
    # ax.set_xlim(590, 750)
    #%%
    fig, ax = plt.subplots(1, 2, figsize=(20, 10),sharey=True)

    pclip=99.5
    clipVal = np.percentile(np.absolute(data_diff), pclip)
    # Vx
    ax[0].imshow(data_diff, 
                extent=[0, data_diff.shape[1], das_time[-1], das_time[0]],
                aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'))

    ax[0].set_xlabel("Channel number")
    ax[0].set_ylabel("Time [s]")
    ax[0].grid()
    ax[0].set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}')


    ax[1].plot(mag_estimate[:, ::10], das_time, '-k', linewidth=0.1, alpha=0.01)
    ax[1].plot(median_mag, das_time, '--r', linewidth=2, alpha=0.5, zorder=3)
    ax[1].plot(mean_mag, das_time, '-r', linewidth=2, alpha=0.5, zorder=3)
    # ax.vlines(x=[P_arrival_approx, S_arrival_approx], ymax=10, ymin=0)
    ax[1].vlines(x=[eq_mag], ymin=-10, ymax=das_time[-1])
    ax[1].set_xlim(1, 7)
    ax[1].invert_yaxis
    ax[1].set_ylabel('Time (s)')
    ax[1].set_xlabel('Estimated Magnitude')
    plt.savefig(fig_output_dir + f'/{test_event_id}_estimated_mag_vs_DAS.png')

    plt.close('all')
    # ax.set_ylim(0, eq_mag.values*1.4)
    # ax.set_xlim(590, 750)
    # %%
