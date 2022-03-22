#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# import the plotting functions
from plotting_functions import *
# import the utility functions
from utility_functions import *


# Use the predicted strain to calculate magnitude
def calculate_magnitude_from_strain(peak_amplitude_df, reg, fitting_type='without_site', site_term_column='region_site'):
    
    if fitting_type == 'with_site':
        # get the annoying categorical keys
        try:
            site_term_keys = np.array([f'C({site_term_column})[{site_term}]' for site_term in peak_amplitude_df[site_term_column]])
        except:
            raise NameError(f"Index {site_term_column} doesn't exist!")
            
        M_predict = (np.log10(peak_amplitude_df.peak_P) \
                    - np.array(reg.params[site_term_keys]) \
                    - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                    / reg.params['magnitude']
        
    elif fitting_type == 'without_site':
        M_predict = (np.log10(peak_amplitude_df.peak_P) \
                    - reg.params['Intercept'] \
                    - reg.params['np.log10(distance_in_km)'] * np.log10(peak_amplitude_df.distance_in_km)) \
                    / reg.params['magnitude']
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


# First check how well the regression relation can be used to calculate Magnitude

#%% load the results from combined regional site terms t
results_output_dir = '/kuafu/yinjx/combined_strain_scaling'
regression_dir = 'regression_results_smf'

nearby_channel_numbers = [100, 50, 20, 10]
horizontal_shift = [-0.045, -0.015, 0.015, 0.045]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

plt.close('all')
fig, ax = plt.subplots(2,1, figsize=(8, 16), sharex=True, sharey=True)

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_region_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_region_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='region_site')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='region_site')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)
    
    #ax[0].scatter(peak_amplitude_df.magnitude + horizontal_shift[ii], M_P, s=5, c=cmap[ii], marker='.', alpha=0.05, label=f'{nearby_channel_number} channels')
    ax[0].errorbar(temp_df_P.magnitude + horizontal_shift[ii], temp_df_P.predicted_M, yerr=temp_df_P.predicted_M_std, marker='.', color=cmap[ii])
    ax[0].scatter(temp_df_P.magnitude + horizontal_shift[ii], temp_df_P.predicted_M,s=20, c=cmap[ii], 
    marker='o', edgecolors='k', label=f'{nearby_channel_number} channels mean', zorder=2)
    ax[0].plot([0, 10], [0, 10], '-k', zorder=1)
    ax[0].set_xlim(1.5, 6.5)
    ax[0].set_ylim(1.5, 6.5)

    #ax[1].scatter(peak_amplitude_df.magnitude + horizontal_shift[ii], M_S, s=5, c=cmap[ii], marker='.', alpha=0.05, label=f'{nearby_channel_number} channels')
    ax[1].errorbar(temp_df_S.magnitude + horizontal_shift[ii], temp_df_S.predicted_M, yerr=temp_df_S.predicted_M_std, marker='.', color=cmap[ii])
    ax[1].scatter(temp_df_S.magnitude + horizontal_shift[ii], temp_df_S.predicted_M,s=20, c=cmap[ii], 
    marker='o', edgecolors='k', label=f'{nearby_channel_number} channels mean', zorder=2)
    ax[1].plot([0, 10], [0, 10], '-k', zorder=1)

ax[0].legend(loc=4)
ax[1].legend(loc=4)

#%% load the results from Ridgecrest
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
regression_dir = 'regression_results_smf'

for nearby_channel_number in [10]:#[10, 20, 50, 100]:
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   

# %%
M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')
# %%
temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

fig, ax = plt.subplots(2,1, figsize=(8, 16), sharex=True, sharey=True)
ax[0].scatter(temp_df_P.magnitude, temp_df_P.predicted_M,s=10, c='r', marker='.')
ax[0].plot([0, 10], [0, 10], '-k', zorder=1)
ax[0].set_xlim(2, 6)
ax[0].set_ylim(2, 6)

ax[1].scatter(temp_df_S.magnitude, temp_df_S.predicted_M,s=10, c='r', marker='.')
ax[1].plot([0, 10], [0, 10], '-k', zorder=1)


#%% load the results from Olancha
results_output_dir = '/home/yinjx/kuafu/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain_rate'
regression_dir = 'regression_results_smf'

for nearby_channel_number in [10]:#[10, 20, 50, 100]:
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   

# %%
M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')
# %%
temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

fig, ax = plt.subplots(2,1, figsize=(8, 16), sharex=True, sharey=True)
ax[0].scatter(temp_df_P.magnitude, temp_df_P.predicted_M,s=10, c='r', marker='.')
ax[0].plot([0, 10], [0, 10], '-k', zorder=1)
ax[0].set_xlim(2, 6)
ax[0].set_ylim(2, 6)

ax[1].scatter(temp_df_S.magnitude, temp_df_S.predicted_M,s=10, c='r', marker='.')
ax[1].plot([0, 10], [0, 10], '-k', zorder=1)
# %%
