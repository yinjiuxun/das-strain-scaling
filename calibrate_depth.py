#%% import modules
import os
from cv2 import normalize
import pandas as pd
#from sep_util import read_file
import numpy as np
import matplotlib.pyplot as plt

# Plotting
import matplotlib
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
das_pick_file_folder_list = ['/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events',
                            '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events',
                            '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events',
                            '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events']

catalog_file_list = ['/kuafu/EventData/Ridgecrest/catalog.csv',
                '/kuafu/EventData/Mammoth_south/catalog.csv',
                '/kuafu/EventData/Mammoth_north/catalog.csv',
                '/kuafu/EventData/Sanriku_ERI/catalog.csv']

region_list = ['Ridgecrest', 'Long-Vally south', 'Long-Valley north', 'Sanriku']

das_pick_file_name = 'peak_amplitude.csv'

#%%
for ii in [3]:#range(4):
    das_pick_file_folder = das_pick_file_folder_list[ii]
    catalog_file = catalog_file_list[ii]
    print(f'=========== Working on {das_pick_file_folder} ===============')

    peak_amplitude_df = pd.read_csv(das_pick_file_folder + '/' + das_pick_file_name)
    catalog = pd.read_csv(catalog_file)

    # combine the two dataframe to include the event depth information
    catalog_depth = catalog.loc[:, ['event_id', 'depth_km']]
    peak_amplitude_df = pd.merge(peak_amplitude_df, catalog_depth, on="event_id")
    peak_amplitude_df['calibrated_distance_in_km'] = np.sqrt(peak_amplitude_df.depth_km**2 + peak_amplitude_df.distance_in_km**2)

    if ii==3:
        temp1 = np.array([0, 1, 25, 2, 3, 26])
        temp2 = np.arange(4, 25)
    else:
        temp1 = np.array([0, 1, 24, 2, 3, 25])
        temp2 = np.arange(4, 24)
    
    reorder_index = np.concatenate((temp1, temp2), axis=0)
    peak_amplitude_df = peak_amplitude_df.iloc[:, reorder_index]

    peak_amplitude_df.to_csv(das_pick_file_folder + '/calibrated_' + das_pick_file_name, index=False)

# %%
fig, ax = plt.subplots(2,2, figsize=(12, 12), sharex=True, sharey=True)
ax = ax.flatten()
for ii in range(4):
    das_pick_file_folder = das_pick_file_folder_list[ii]
    catalog_file = catalog_file_list[ii]
    print(f'=========== Working on {das_pick_file_folder} ===============')

    peak_amplitude_df = pd.read_csv(das_pick_file_folder + '/calibrated_' + das_pick_file_name)
    catalog = pd.read_csv(catalog_file)

    gca = ax[ii]
    gca.hist(np.log10(peak_amplitude_df.distance_in_km), alpha=0.5, range=(0, 3.5), bins=50, density=True, label='epicenter distance')
    gca.hist(np.log10(peak_amplitude_df.calibrated_distance_in_km), alpha=0.5, range=(0, 3.5), bins=50, density=True, label='hypocenter distance')
    gca.set_ylabel('PDF')
    gca.set_xlabel('distance (km) in log10')
    gca.set_title(region_list[ii])

gca.legend()
plt.savefig('depth_calibrated_distance_distribution.png', bbox_inches='tight')
# %%
