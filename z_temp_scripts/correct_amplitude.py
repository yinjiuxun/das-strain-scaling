#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import time

sys.path.append('../')
from utility.general import *
from utility.loading import *

import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

#%%
event_dir = '/kuafu/EventData/Ridgecrest'
catalog = pd.read_csv(event_dir + '/catalog.csv')
all_files = glob.glob(event_dir + '/data/*.h5')


# files before this time need amplitude correction
time_change = os.path.getmtime(all_files[200])-30*3600*24

# Example of correcting amplitude

def correct_amplitude(event_dir, catalog, time_change, file):
    if os.path.getmtime(file) < time_change:
        # correct the amplitude for the Olancha new data
        event_id = int(file[-11:-3])
        data, info = load_event_data(event_dir + '/data/', event_id)
        # correct amplitude
        data *= 8/16.35 
        # correct the info
        info['begin_time'] = str(info['begin_time'])
        info['end_time'] = str(info['end_time'])
        info['event_time'] = str(info['event_time'])
        info['magnitude_type'] = catalog[catalog.event_id == event_id].magnitude_type.iloc[0]

        # write the corrected info and data back to the h5 files
        save_rawevent_h5(event_dir + f'/data/{event_id}.h5', data, info)
        # print(file + '   done!')

#%%
with tqdm_joblib(tqdm(desc="Correct amplitude", total=len(all_files))) as progress_bar:
    Parallel(n_jobs=Ncores)(delayed(correct_amplitude)(event_dir, catalog, time_change, file) for file in all_files)


# %%




# for file in all_files:
#     # correct the amplitude for the Olancha new data
#     event_id = int(file[-11:-3])
#     data, info = load_event_data(event_dir + '/data/', event_id)

#     # # correct amplitude
#     # data /= 1e6 

#     # correct the info
#     info['begin_time'] = str(info['begin_time'])
#     info['end_time'] = str(info['end_time'])
#     info['event_time'] = str(info['event_time'])
#     info['magnitude_type'] = catalog[catalog.event_id == event_id].magnitude_type.iloc[0]

#     # write the corrected info and data back to the h5 files
#     save_rawevent_h5(event_dir + f'/data/{event_id}.h5', data, info)
#     print(file + '   done!')



#%%
# Example of correcting amplitude
# event_dir = '/kuafu/EventData/Olancha_new'
# catalog = pd.read_csv(event_dir + '/catalog.csv')
# all_files = glob.glob(event_dir + '/data/*.h5')

# for file in all_files:
#     # correct the amplitude for the Olancha new data
#     event_id = int(file[-11:-3])
#     data, info = load_event_data(event_dir + '/data/', event_id)
#     # correct amplitude
#     data /= 1e6 
#     # correct the info
#     info['begin_time'] = str(info['begin_time'])
#     info['end_time'] = str(info['end_time'])
#     info['event_time'] = str(info['event_time'])
#     info['magnitude_type'] = catalog[catalog.event_id == event_id].magnitude_type.iloc[0]

#     # write the corrected info and data back to the h5 files
#     ## save_rawevent_h5(event_dir + f'/data/{event_id}.h5', data, info)
#     print(file + '   done!')




















# # %%
# # Combine all the individual peak amplitude files into one for regression
# event_folder = '/kuafu/EventData/Ridgecrest'
# peak_amplitude_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events'

# print('='*10 + peak_amplitude_dir + '='*10)
# temp = glob.glob(peak_amplitude_dir + '/*.csv')
   
# # %%
# # function to correct
# def correct_amplitude(csv_file, correction_factor):
#     peak_df = pd.read_csv(csv_file)
#     peak_df['peak_P'] *= correction_factor
#     peak_df['peak_P_1s'] *= correction_factor
#     peak_df['peak_P_3s'] *= correction_factor
#     peak_df['peak_P_4s'] *= correction_factor
#     peak_df['peak_S'] *= correction_factor
#     peak_df['peak_S_4s'] *= correction_factor
#     peak_df['peak_S_6s'] *= correction_factor
#     peak_df['peak_S_8s'] *= correction_factor
#     peak_df['peak_S_10s'] *= correction_factor
#     peak_df.to_csv(csv_file, index=False)
# #%%
# # run correction
# n_files = len(temp)
# correction_factor = 8/16.35
# with tqdm_joblib(tqdm(desc="Correct amplitude", total=n_files)) as progress_bar:
#     Parallel(n_jobs=Ncores)(delayed(correct_amplitude)(csv_file, correction_factor) for csv_file in temp)

