#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append('../')
from utility.general import *

import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

#%%
# correct the site terms






































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

