#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

# import the plotting functions
from plotting_functions import *
from utility_functions import *

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# %%
# ==============================  Looking into the multiple array case ========================================
# Specify the file names
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'

# directory to store the fitted results:
regression_results_dir = results_output_dir + '/regression_results_smf_weighted_100_channel_at_least'
# %% Compare the regression parameters and site terms


title_plot = ['Ridgecrest P', 'Ridgecrest S', 'Long Valley North P', 'Long Valley North S', 'Long Valley South P', 'Long Valley South S']

combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{combined_channel_number}.csv')
    region_site = np.sort(peak_amplitude_df.region_site.unique())

    temp_df_P = pd.DataFrame(columns=['region_site', 'site_term_P'])
    temp_df_S = pd.DataFrame(columns=['region_site', 'site_term_S'])

    try:
        regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
        site_term_P = regP.params[:-2]
        # temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
        temp_df_P['region_site'] = [temp.replace('C(region_site)[', '').replace(']', '') for temp in site_term_P.index]
        temp_df_P['site_term_P'] = np.array(site_term_P)
        #temp_df_P = get_site_term_dataframe(temp_df_P)
    except:
        print('P regression not found, assign Nan to the site term')
        site_term_P = np.nan
    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
        site_term_S = regS.params[:-2]
        # temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])
        temp_df_S['region_site'] = [temp.replace('C(region_site)[', '').replace(']', '') for temp in site_term_S.index]
        temp_df_S['site_term_S'] = np.array(site_term_S)
        #temp_df_S = get_site_term_dataframe(temp_df_S)
    except:
        print('S regression not found, assign Nan to the site term')
        site_term_S = np.nan

    temp_df1 = pd.DataFrame(columns=['region_site'])
    temp_df1['region_site'] = region_site
    temp_df1 = pd.merge(temp_df1, temp_df_P, on='region_site', how='outer')
    temp_df1 = pd.merge(temp_df1, temp_df_S, on='region_site', how='outer')

    temp_df2 = peak_amplitude_df.loc[:, ['region_site', 'channel_id']]
    site_term_df = pd.merge(temp_df2, temp_df1, on='region_site', how='outer')
    site_term_df = site_term_df.drop_duplicates(subset=['region_site', 'channel_id'])
    site_term_df.to_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv', index=False)
    
# %%
