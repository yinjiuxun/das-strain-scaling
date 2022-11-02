#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# import the plotting functions
from plotting_functions import *


#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_name = '/Mammoth_South_Scaling_M3.csv'

# ==============================  Mammoth data - North========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_name = '/Mammoth_North_Scaling_M3.csv'

# ==============================  Ridgecrest data ========================================
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate_refined'
das_pick_file_name = '/peak_amplitude_M3+.csv'
region_label = 'ridgecrest'

peak_amplitude_df = pd.read_csv(results_output_dir + '/' + das_pick_file_name)
# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling'
das_pick_file_name = '/peak_amplitude_M3+.csv'
region_label = 'olancha'
peak_amplitude_df2 = pd.read_csv(results_output_dir + '/' + das_pick_file_name)


#%%
plt.loglog(peak_amplitude_df.distance_in_km, peak_amplitude_df.peak_P, 'rx', alpha=0.01)
plt.loglog(peak_amplitude_df2.distance_in_km, peak_amplitude_df2.peak_P, 'bx', alpha=0.01)

#%%
plt.semilogy(peak_amplitude_df.magnitude, peak_amplitude_df.peak_P, 'rx', alpha=0.01)
plt.semilogy(peak_amplitude_df2.magnitude, peak_amplitude_df2.peak_P, 'bx', alpha=0.01)

#%% load the peak amplitude results
# Load the peak amplitude results
peak_amplitude_df = pd.read_csv(results_output_dir + '/' + das_pick_file_name)

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)


peak_amplitude_df = peak_amplitude_df.dropna()
DAS_index = peak_amplitude_df.channel_id.unique().astype('int')
# %%
peak_amplitude_df.peak_P = peak_amplitude_df.peak_P * 1e6
peak_amplitude_df.peak_S = peak_amplitude_df.peak_S * 1e6
peak_amplitude_df.peak_P_strain = peak_amplitude_df.peak_P_strain * 1e6
peak_amplitude_df.peak_S_strain = peak_amplitude_df.peak_S_strain * 1e6


# %%
peak_amplitude_df.to_csv(results_output_dir + '/' + das_pick_file_name, index=False)
# %%


#%% Specify the file names
DAS_info_files = '/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS_Olancha_Plexus.txt'
catalog_file =  '/home/yinjx/Olancha_tap_test/Olancha_tap_test.txt'
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

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf'
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

# %%Find events in the pick file
catalog_select = catalog[(catalog[7]>2) & (catalog[4]>36.2)]
num_events = catalog_select.shape[0]
catalog_select.columns=['id', 'type', ' ', 'time', 'lat', 'lon', 'depth', 'mag', 'MT', 'qual']
# %%
plt.plot(DAS_lon,DAS_lat, 'k.')
plt.plot(catalog_select['lon'],catalog_select['lat'], 'o')
plt.savefig('/home/yinjx/Olancha_tap_test/simple_map.png')
# %%
catalog_select.to_csv('/home/yinjx/Olancha_tap_test/Olancha_tap_test_selected_events.txt', index=False)
# %%




import obspy
# %%
tr = obspy.read('/home/yinjx/39462536/*AZ.BZN*.sac')
# %%
tr.plot()
plt.show()
# %%
