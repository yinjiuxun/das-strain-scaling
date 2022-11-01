#%%
# Import modules
import pandas as pd
#from sep_util import read_file
import numpy as np
import glob



event_folder_list = ['/kuafu/EventData/Ridgecrest', '/kuafu/EventData/Mammoth_north', '/kuafu/EventData/Mammoth_south']
peak_amplitude_dir_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events', 
                           '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events']
for ii_region in [1, 2]:
    peak_amplitude_dir = peak_amplitude_dir_list[ii_region]

    print('='*10 + peak_amplitude_dir + '='*10)

    temp = glob.glob(peak_amplitude_dir + '/*.csv')
    temp_df = pd.concat(map(pd.read_csv, temp), ignore_index=True)
    temp_df.to_csv(peak_amplitude_dir + '/peak_amplitude.csv', index=False)
# %%
