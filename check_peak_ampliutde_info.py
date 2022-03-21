#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np

# import the plotting functions
from plotting_functions import *
# import the utility functions
from utility_functions import *

# ==============================  Ridgecrest data ========================================
#%% Specify the file names
DAS_info_files = '/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Ridgecrest_ODH3.txt'
catalog_file =  '/home/yinjx/notebooks/strain_scaling/Ridgecrest_das_catalog_M2_M8.txt'
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_name = '/peak_amplitude_M3+.csv'

# ==============================  Olancha data ========================================
#%% Specify the file names
DAS_info_files = '/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Olancha_Plexus.txt'
catalog_file =  '/home/yinjx/notebooks/strain_scaling/Ridgecrest_das_catalog_M2_M8.txt'
results_output_dir = '/kuafu/yinjx/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_name = '/peak_amplitude_M3+.csv'



#%% load the information about the DAS channel
DAS_info = np.genfromtxt(DAS_info_files)
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info[:, 0].astype('int')
DAS_lon = DAS_info[:, 1]
DAS_lat = DAS_info[:, 2]

# load the catalog
catalog = pd.read_csv(catalog_file, sep='\s+', header=None, skipfooter=1, engine='python')

# Load the peak amplitude results
peak_amplitude_df = pd.read_csv(results_output_dir + '/' + das_pick_file_name)
# Add the event label for plotting
peak_amplitude_df = add_event_label(peak_amplitude_df)

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

# Find events in the pick file
event_id_selected = np.unique(peak_amplitude_df['event_id'])
catalog_select = catalog[catalog[0].isin(event_id_selected)]
num_events = catalog_select.shape[0]
catalog_select

#%% Show the basic information about the array and events
plot_time_variation_of_events(catalog_select, results_output_dir + '/time_variation_selected_earthquakes.png')

plot_simple_map_of_das_and_events(DAS_index, DAS_lon, DAS_lat, catalog_select, results_output_dir + '/map_of_earthquakes_not_grouped.png')

# %% Show the peak strain rate amplitude variations
plot_magnitude_distance_coverage(peak_amplitude_df, results_output_dir + '/magnitude_distance_distribution.png')
plot_distance_variations(peak_amplitude_df, ['peak_P', 'peak_S'], results_output_dir + '/peak_strain_rate_vs_distance.png')
plot_magnitude_variations(peak_amplitude_df, ['peak_P', 'peak_S'], results_output_dir + '/peak_strain_rate_vs_magnitude.png')
plot_P_S_amplitude_ratio(peak_amplitude_df, ['peak_P', 'peak_S'], results_output_dir + '/peak_strain_rate_P_S_ratio.png')

# %% Show the peak strain amplitude variations
plot_distance_variations(peak_amplitude_df, ['peak_P_strain', 'peak_S_strain'], results_output_dir + '/peak_strain_vs_distance.png')
plot_magnitude_variations(peak_amplitude_df, ['peak_P_strain', 'peak_S_strain'], results_output_dir + '/peak_strain_vs_magnitude.png')
plot_P_S_amplitude_ratio(peak_amplitude_df, ['peak_P_strain', 'peak_S_strain'], results_output_dir + '/peak_strain_P_S_ratio.png')
# %%


