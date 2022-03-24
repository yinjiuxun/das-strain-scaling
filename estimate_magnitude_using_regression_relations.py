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

#%% Define functions
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

def plot_magnitude_prediction(temp_df_P_list, temp_df_S_list):
    horizontal_shift = [0, 0, 0, 0] #[-0.015, -0.005, 0.005, 0.015]
    fig, ax = plt.subplots(2, 1, figsize=(10, 20), sharex=True, sharey=True)
    for ii in range(len(temp_df_P_list)):
        temp_df_P = temp_df_P_list[ii]
        temp_df_S = temp_df_S_list[ii]

        ax[0].errorbar(temp_df_P.magnitude + horizontal_shift[ii], temp_df_P.predicted_M, yerr=temp_df_P.predicted_M_std, marker='o', linestyle='none')
        ax[0].plot([0, 10], [0, 10], '-k', zorder=1)
        ax[0].vlines(x=4, ymin=0, ymax=10, linestyle='--', color='k')
        ax[0].set_xlim(2, 6)
        ax[0].set_ylim(2, 6)
        ax[0].set_ylabel('P predicted M')
        ax[0].set_xlabel('true M')
        ax[0].xaxis.set_tick_params(which='both',labelbottom=True)

        ax[1].errorbar(temp_df_S.magnitude + horizontal_shift[ii], temp_df_S.predicted_M, yerr=temp_df_S.predicted_M_std, marker='o', linestyle='none')
        ax[1].plot([0, 10], [0, 10], '-k', zorder=1)
        ax[1].vlines(x=4, ymin=0, ymax=10, linestyle='--', color='k')
        ax[1].set_xlim(2, 6.5)
        ax[1].set_ylim(2, 6.5)
        ax[1].set_ylabel('S predicted M')
        ax[1].set_xlabel('true M')

    ax[0].text(2.5, 5.75, 'regression')
    ax[0].text(4.5, 5.75, 'prediction')
    ax[1].text(2.5, 5.75, 'regression')
    ax[1].text(4.5, 5.75, 'prediction')

# ========================== work on Combined results from both Ridgecrest and Olancha ================================
# First check how well the regression relation can be used to calculate Magnitude
#%% load the results from combined regional site terms t
results_output_dir = '/kuafu/yinjx/combined_strain_scaling'
regression_dir = 'regression_results_smf'

nearby_channel_numbers = [100, 50, 20, 10]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

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

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)
#%%
plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")

# Then try to use the regression relation from small events to predict the larger ones
#%% load the results from combined regional site terms t
results_output_dir = '/kuafu/yinjx/combined_strain_scaling'
regression_dir = 'regression_results_smf_M3'

nearby_channel_numbers = [100, 50, 20, 10]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

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

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)
#%%
plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")



#%% ========================== work on the results from Ridgecrest ================================
# First check how well the regression relation can be used to calculate Magnitude
#%% load the results from combined regional site terms t
results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
regression_dir = 'regression_results_smf'

nearby_channel_numbers = [100, 50, 20, 10]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")

# Then try to use the regression relation from small events to predict the larger ones
# load the results from combined regional site terms t
regression_dir = 'regression_results_smf_M3'

nearby_channel_numbers = [100, 50, 20, 10]
horizontal_shift = [-0.015, -0.005, 0.005, 0.015]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")



#%% ========================== work on the results from Olancha ================================
# First check how well the regression relation can be used to calculate Magnitude
#%% load the results from combined regional site terms t
results_output_dir = '/home/yinjx/kuafu/Olancha_Plexus/Olancha_scaling/peak_ampliutde_scaling_results_strain_rate'
regression_dir = 'regression_results_smf'

nearby_channel_numbers = [100, 50, 20, 10]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")

# Then try to use the regression relation from small events to predict the larger ones
# load the results from combined regional site terms t
regression_dir = 'regression_results_smf_M3'

nearby_channel_numbers = [100, 50, 20, 10]
horizontal_shift = [-0.015, -0.005, 0.005, 0.015]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")


#%% ========================== work on the results from Mammoth South ================================
# First check how well the regression relation can be used to calculate Magnitude
#%% load the results from combined regional site terms t
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
regression_dir = 'regression_results_smf'

nearby_channel_numbers = [100, 50, 20, 10]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")

# Then try to use the regression relation from small events to predict the larger ones
# load the results from combined regional site terms t
regression_dir = 'regression_results_smf_M4'

nearby_channel_numbers = [100, 50, 20, 10]
horizontal_shift = [-0.015, -0.005, 0.005, 0.015]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")

#%% ========================== work on the results from Mammoth North ================================
# First check how well the regression relation can be used to calculate Magnitude
#%% load the results from combined regional site terms t
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
regression_dir = 'regression_results_smf'

nearby_channel_numbers = [100, 50, 20, 10]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")

# Then try to use the regression relation from small events to predict the larger ones
# load the results from combined regional site terms t
regression_dir = 'regression_results_smf_M4'

nearby_channel_numbers = [100, 50, 20, 10]
horizontal_shift = [-0.015, -0.005, 0.005, 0.015]
cmap = ['blue', 'orange', 'green', 'red', 'purple', 'yellow']

temp_df_P_list = []
temp_df_S_list = []

for ii, nearby_channel_number in enumerate(nearby_channel_numbers):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
    

    # %% Now can fit the data with different regional site terms
    regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")
    regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.pickle")

    print(f'Combined every {nearby_channel_number} channels.')
    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')   


    M_P = calculate_magnitude_from_strain(peak_amplitude_df, regP, fitting_type='with_site', site_term_column='combined_channel_id')
    M_S = calculate_magnitude_from_strain(peak_amplitude_df, regS, fitting_type='with_site', site_term_column='combined_channel_id')

    temp_df_P = get_mean_magnitude(peak_amplitude_df, M_P)
    temp_df_S = get_mean_magnitude(peak_amplitude_df, M_S)

    temp_df_P_list.append(temp_df_P)
    temp_df_S_list.append(temp_df_S)

plot_magnitude_prediction(temp_df_P_list, temp_df_S_list)
plt.savefig(results_output_dir + '/' + regression_dir + "/predicted_magnitude.png")
# %%
