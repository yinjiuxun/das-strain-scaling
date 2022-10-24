#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

from utility.general import mkdir

import matplotlib
import matplotlib.pyplot as plt

from utility.processing import filter_event
from utility.processing import filter_event

# set plotting parameters 
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 100,  # to adjust notebook inline plot size
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

from utility.plotting import add_annotate
#%%
# plot site terms from all-region regression
# some parameters
snr_threshold = 10
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [0, 10]

region_y_minP, region_y_maxP = [0, -1, -1], [1.5, 1.5, 1.5]
region_y_minS, region_y_maxS = [0, -1, -1, -1.5], [1.5, 1.5, 1.5, 0]

results_output_dir_list = []
dx_list = []

#%%
#  # Set result directory
# multiple arrays
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
region_list = ['ridgecrest', 'mammothN', 'mammothS']
region_text = ['Ridgecrest', 'Long Valley (N)', 'Long Valley (S)']
site_term_list = ['P', 'S']
dx_list = [8, 10, 10] # Ridgecrest, Mammoth N, Mammoth S


regression_results_dir = results_output_dir + f'/iter_regression_results_smf_weighted_{min_channel}_channel_at_least'
site_term_df = pd.read_csv(regression_results_dir + '/site_terms_iter.csv')


fig, ax = plt.subplots(3, 2, figsize=(16, 12), )
for i_region in range(len(region_list)):
    region_key, region_dx = region_list[i_region], dx_list[i_region]

    for i_wavetype, wavetype in enumerate(site_term_list):
        temp = site_term_df[site_term_df.region == region_key]
        gca = ax[i_region, i_wavetype]
        gca.plot(temp.channel_id*region_dx/1e3, temp[f'site_term_{wavetype.upper()}'], '-k')
        gca.set_title(f'{region_text[i_region]}, site term of {wavetype.upper()} wave')

        if i_wavetype == 0:
            gca.set_ylim(region_y_minP[i_region], region_y_maxP[i_region])
        elif i_wavetype == 1:
            gca.set_ylim(region_y_minS[i_region], region_y_maxS[i_region])
        else:
            raise

        if i_region == 2:
            gca.set_xlabel('Distance to IU (km)')

        if i_wavetype == 0:
            gca.set_ylabel('Site term in log10')

ax = add_annotate(ax)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
mkdir(regression_results_dir + '/figures')
plt.savefig(regression_results_dir + '/figures/site_terms.png', bbox_inches='tight')

# %%
# single arrays
results_output_dir_list = []
region_key_list = []
region_text_list = []
region_dx_list = []

# Ridgecrest
results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
region_dx = 8
region_key = 'ridgecrest'
region_text = 'Ridgecrest'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# Long Valley N
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
region_dx = 10
region_key = 'mammothN'
region_text = 'Long Valley (N)'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# Long Valley S
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
region_dx = 10
region_key = 'mammothS'
region_text = 'Long Valley (S)'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# Sanriku 
results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
region_dx = 5
region_key = 'sanriku'
region_text = 'Sanriku'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# LAX 
results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
region_dx = 10
region_key = 'LA-Google'
region_text = 'LA-Google'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# some special locations along the cable
bridge_section = np.arange(1940, 1954)*region_dx/1e3
metro_section = np.concatenate([range(4050, 4201), range(4250, 4351), range(4450, 4651)]) * region_dx/1e3

#%%
weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise

for i_region, results_output_dir in enumerate(results_output_dir_list):
    print(results_output_dir)
    region_key = region_key_list[i_region]
    region_text = region_text_list[i_region]
    region_dx = region_dx_list[i_region]


    regression_results_dir = results_output_dir + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    site_term_df = pd.read_csv(regression_results_dir + '/site_terms_iter.csv')

    # load the peak amplitude file
    peak_amplitude_file = results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv'
    peak_amplitude_df_fit = pd.read_csv(peak_amplitude_file)

    site_term_list = ['P', 'S']
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for i_wavetype, wavetype in enumerate(site_term_list):
        temp = site_term_df[site_term_df.region == region_key]

        
        peak_amplitude_df_temp = filter_event(peak_amplitude_df_fit, M_threshold, snr_threshold, min_channel)
        peak_amplitude_df_temp = peak_amplitude_df_temp[~peak_amplitude_df_temp[f'peak_{wavetype.upper()}'].isna()]
        
        gca = ax[0, i_wavetype]
        try:
            gca.hist(peak_amplitude_df_temp.channel_id*region_dx/1e3, range=(0.5*region_dx/1e3, (peak_amplitude_df_temp.channel_id.max()+0.5)*region_dx/1e3), bins=int(peak_amplitude_df_temp.channel_id.max()), color='k')
        except:
            print(f"No {wavetype}, skip...")
        gca.set_ylabel('Number of measurements')
        
        gca = ax[1, i_wavetype]
        try:
            #gca.plot(temp.channel_id*region_dx/1e3, temp[f'site_term_{wavetype.upper()}'], 'k.')
            gca.scatter(temp.channel_id*region_dx/1e3, temp[f'site_term_{wavetype.upper()}'],s=1, c=temp[f'site_term_{wavetype.upper()}'], cmap='viridis')
        except:
            print(f"No {wavetype}, skip...")

        # if region_text == 'LA-Google':
        #     gca.vlines(x=bridge_section[[0, -1]], ymin=np.nanmin(temp[f'site_term_{wavetype.upper()}']), ymax=np.nanmax(temp[f'site_term_{wavetype.upper()}']), color='blue', label='Bridge')
        #     gca.vlines(x=metro_section[[0, -1]], ymin=np.nanmin(temp[f'site_term_{wavetype.upper()}']), ymax=np.nanmax(temp[f'site_term_{wavetype.upper()}']), color='red', label='Metro')

        # gca.set_xlim(metro_section[0]-0.5, metro_section[-1]+0.5)
        gca.set_title(f'{region_text}, site term of {wavetype.upper()} wave')
        gca.set_xlabel('Distance to IU (km)')
        
        gca.set_ylabel('Site term in log10')


    if region_text == 'LA-Google':
        ax[0, 0].set_yticks(range(0, 100, 5))
        ax[0, 0].set_ylim(0, 15)

    ax = add_annotate(ax)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    mkdir(regression_results_dir + '/figures')
    plt.savefig(regression_results_dir + '/figures/site_terms.png', bbox_inches='tight')


#%% 
# Compare the site terms from multi arrays and single arrays
#  # Set result directory
# multiple arrays
results_output_dir_multi = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
region_list = ['ridgecrest', 'mammothN', 'mammothS']
region_text = ['Ridgecrest', 'Long Valley (N)', 'Long Valley (S)']
site_term_list = ['P', 'S']
dx_list = [8, 10, 10] # Ridgecrest, Mammoth N, Mammoth S
region_y_minP, region_y_maxP = [0, -1, -1], [1.5, 1.5, 1.5]
region_y_minS, region_y_maxS = [0, -1, -1, -1.5], [1.5, 1.5, 1.5, 0]

# single arrays
results_output_dir_list = []
region_key_list = []
region_text_list = []
region_dx_list = []

# Ridgecrest
results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
region_dx = 8
region_key = 'ridgecrest'
region_text = 'Ridgecrest'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# Long Valley N
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
region_dx = 10
region_key = 'mammothN'
region_text = 'Long Valley (N)'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# Long Valley S
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
region_dx = 10
region_key = 'mammothS'
region_text = 'Long Valley (S)'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)


# Show comparison
regression_results_dir_multi = results_output_dir_multi + f'/iter_regression_results_smf_weighted_{min_channel}_channel_at_least'
site_term_df_multi = pd.read_csv(regression_results_dir_multi + '/site_terms_iter.csv')

fig, ax = plt.subplots(3, 2, figsize=(16, 12), )
for i_region, results_output_dir_single in enumerate(results_output_dir_list):
    region_key, region_dx = region_list[i_region], dx_list[i_region]

    regression_results_dir_single = results_output_dir_single + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    site_term_df_single = pd.read_csv(regression_results_dir_single + '/site_terms_iter.csv')

    for i_wavetype, wavetype in enumerate(site_term_list):
        temp_multi = site_term_df_multi[site_term_df_multi.region == region_key]
        temp_single = site_term_df_single[site_term_df_single.region == region_key]

        gca = ax[i_region, i_wavetype]
        gca.plot(temp_multi.channel_id*region_dx/1e3, temp_multi[f'site_term_{wavetype.upper()}'], '-k', label='all arrays')
        gca.plot(temp_single.channel_id*region_dx/1e3, temp_single[f'site_term_{wavetype.upper()}'], '-r', label='single array')

        gca.set_title(f'{region_text_list[i_region]}, site term of {wavetype.upper()} wave')

        if i_wavetype == 0:
            gca.set_ylim(region_y_minP[i_region], region_y_maxP[i_region])
        elif i_wavetype == 1:
            gca.set_ylim(region_y_minS[i_region], region_y_maxS[i_region])
        else:
            raise

        if i_region == 2:
            gca.set_xlabel('Distance to IU (km)')

        if i_wavetype == 0:
            gca.set_ylabel('Site term in log10')

        if (i_wavetype == 0) and (i_region ==0):
            gca.legend()

ax = add_annotate(ax)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
mkdir(regression_results_dir + '/figures')
plt.savefig(regression_results_dir_multi + '/figures/site_terms_compare.png', bbox_inches='tight')



















#%% The Following will be removed soon!


#%%
# some parameters
snr_threshold = 10
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [0, 10]

# # Set result directory
# multiple arrays
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
region_list = ['ridgecrest', 'mammothN', 'mammothS']
region_text = ['Ridgecrest', 'Long Valley (N)', 'Long Valley (S)']
site_term_list = ['P', 'S']
dx_list = [8, 10, 10] # Ridgecrest, Mammoth N, Mammoth S


regression_results_dir = results_output_dir + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
site_term_df = pd.read_csv(regression_results_dir + '/site_terms_iter.csv')


fig, ax = plt.subplots(3, 2, figsize=(14, 14), )
for i_region in range(len(region_list)):
    region_key, region_dx = region_list[i_region], dx_list[i_region]

    # load results from single
    regression_results_single = results_output_dir_list[i_region] + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    site_term_df_single = pd.read_csv(regression_results_single + '/site_terms_iter.csv')
    
    for i_wavetype, wavetype in enumerate(site_term_list):
        temp = site_term_df[site_term_df.region == region_key]
        gca = ax[i_region, i_wavetype]
        gca.plot(temp.channel_id*region_dx/1e3, temp[f'site_term_{wavetype.upper()}'])
        gca.plot(site_term_df_single.channel_id*region_dx/1e3, site_term_df_single[f'site_term_{wavetype.upper()}'])
        
        gca.set_title(f'{region_text[i_region]}, site term of {wavetype.upper()} wave')

        if i_region == 2:
            gca.set_xlabel('Distance to IU (km)')

        if i_wavetype == 0:
            gca.set_ylabel('Site term in log10')

ax = add_annotate(ax)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.savefig(regression_results_dir + '/figures/site_terms.png', bbox_inches='tight')







from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# ==============================  Ridgecrest data ========================================
#%% Specify the file names
# results_output_dir = '/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_ampliutde_scaling_results_strain_rate'
# das_pick_file_name = '/peak_amplitude_M3+.csv'
# region_label = 'ridgecrest'
min_channel = 100
results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_events'
das_pick_file_name = '/calibrated_peak_amplitude.csv'
regression_results = f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
region_label = 'ridgecrest'
channel_space = 8/1e3
# ==============================  Olancha data ========================================
#%% Specify the file names
results_output_dir = '/kuafu/yinjx/Olancha_Plexus_100km/Olancha_scaling'
das_pick_file_name = '/peak_amplitude_M3+.csv'
region_label = 'olancha'

# ==============================  Mammoth data - South========================================
#%% Specify the file names
min_channel = 100
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events'
das_pick_file_name = '/calibrated_peak_amplitude.csv'
regression_results = f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
region_label = 'mammothS'
channel_space = 10/1e3

# ==============================  Mammoth data - North========================================
#%% Specify the file names
min_channel = 100
results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
das_pick_file_folder = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events'
das_pick_file_name = '/calibrated_peak_amplitude.csv'
regression_results = f'/regression_results_smf_weighted_{min_channel}_channel_at_least'
region_label = 'mammothN'
channel_space = 10/1e3

# ==============================  Sanriku ========================================
#%% Specify the file names
min_channel = 100
results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
das_pick_file_folder = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events'
das_pick_file_name = '/calibrated_peak_amplitude.csv'
regression_results = f'/regression_results_smf_weighted_all_coefficients_drop_4130_{min_channel}_channel_at_least'
region_label = 'Sanriku'
channel_space = 5/1e3
#%% 
# Load the peak amplitude results
snr_threshold = 10
magnitude_threshold = [2, 10]
peak_amplitude_df = pd.read_csv(das_pick_file_folder + '/' + das_pick_file_name)

# directory to store the fitted results
regression_results_dir = results_output_dir + regression_results
if not os.path.exists(regression_results_dir):
    os.mkdir(regression_results_dir)

peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.snrP >=snr_threshold) | (peak_amplitude_df.snrS >=snr_threshold)]
peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >=magnitude_threshold[0]) & (peak_amplitude_df.magnitude <=magnitude_threshold[1])]

if 'QA' in peak_amplitude_df.columns:
    peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.QA == 'Yes']

# peak_amplitude_df = peak_amplitude_df.dropna()
DAS_index = peak_amplitude_df.channel_id.unique().astype('int')

#%% Functions used here
def combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number):
    if nearby_channel_number == -1:
        peak_amplitude_df['combined_channel_id'] = 0
    else:
        temp1= np.arange(0, DAS_index.max()+1) # original channel number
        temp2 = temp1 // nearby_channel_number # combined channel number
        peak_amplitude_df['combined_channel_id'] = temp2[np.array(peak_amplitude_df.channel_id).astype('int')]
    return peak_amplitude_df

def process_region_param_keys(reg, region_string):
    '''A special function to process the regional keys'''
    temp0 = reg.params.keys()
    temp = [region_string in tempx for tempx in temp0]
    site_term_keys = temp0[temp]

    combined_channel = np.array([int(k.replace(f'C(region_site)[{region_string}-','').replace(']','')) for k in site_term_keys])
    ii_sort = np.argsort(combined_channel)

    combined_channel = combined_channel[ii_sort]
    site_term = reg.params[temp].values[ii_sort]

    return combined_channel, site_term

import seaborn as sns
def plot_prediction_vs_measure_seaborn(peak_comparison_df, xy_range, phase):
    sns.set_theme(style="ticks", font_scale=2)
    if phase == 'P':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_P", y="peak_P_predict", marginal_ticks=True,
                        xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    elif phase == 'S':
        g = sns.JointGrid(data=peak_comparison_df, x="peak_S", y="peak_S_predict", marginal_ticks=True,
                        xlim=xy_range, ylim=xy_range, height=10, space=0.3)
    g.ax_joint.set(xscale="log")
    g.ax_joint.set(yscale="log")

# Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.65, .2, .02, .2])

# Add the joint and marginal histogram plots 03012d
    g.plot_joint(sns.histplot, discrete=(False, False), cmap="light:#4D4D9C", pmax=.2, cbar=True, cbar_ax=cax, cbar_kws={'label': 'counts'})
    g.plot_marginals(sns.histplot, element="step", color="#4D4D9C")

    g.ax_joint.plot(xy_range, xy_range, 'k-', linewidth = 2)
    g.ax_joint.set_xlabel('measured peak')
    g.ax_joint.set_ylabel('calculated peak')
    return g

# %% Compare the regression parameters and site terms
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
line_label = ['10 channels', '20 channels', '50 channels', '100 channels', 'constant']
# DataFrame to store parameters for all models
P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'], 
index = np.arange(len(combined_channel_number_list)))
S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
index = np.arange(len(combined_channel_number_list)))

for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
    temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])
    try:
        regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
        site_term_P = regP.params[:-2]
        # temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
        temp_df_P['combined_channel_id'] = [int(temp.replace('C(combined_channel_id)[', '').replace(']', '')) for temp in site_term_P.index]
        temp_df_P['site_term_P'] = np.array(site_term_P)
    except:
        print('P regression not found, assign Nan to the site term')
        site_term_P = np.nan
    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
        site_term_S = regS.params[:-2]
        # temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])
        temp_df_S['combined_channel_id'] = [int(temp.replace('C(combined_channel_id)[', '').replace(']', '')) for temp in site_term_S.index]
        temp_df_S['site_term_S'] = np.array(site_term_S)
    except:
        print('S regression not found, assign Nan to the site term')
        site_term_S = np.nan

    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)
    combined_channel_id = np.sort(peak_amplitude_df.combined_channel_id.unique())

    # combining the site terms into one csv file
    temp_df1 = pd.DataFrame(columns=['combined_channel_id'])
    temp_df1['combined_channel_id'] = combined_channel_id
    temp_df1 = pd.merge(temp_df1, temp_df_P, on='combined_channel_id', how='outer')
    temp_df1 = pd.merge(temp_df1, temp_df_S, on='combined_channel_id', how='outer')

    temp_df2 = peak_amplitude_df.loc[:, ['combined_channel_id', 'channel_id']]
    temp_df2 = temp_df2.drop_duplicates(subset=['channel_id']).sort_values(by='channel_id')
    site_term_df = pd.merge(temp_df2, temp_df1, on='combined_channel_id')
    # Store the site terms
    site_term_df.to_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv', index=False)

# plot to compare the site terms
fig, ax = plt.subplots(1, 2, figsize=(20, 6), sharex=True, sharey=True)
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    try:
        site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')

        ax[0].plot(site_term_df.channel_id*channel_space, site_term_df.site_term_P, '-', label=line_label[i_model])#, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(site_term_df.channel_id*channel_space, site_term_df.site_term_S, '-', label=line_label[i_model])#, Cond.# {regS.condition_number:.2f}')
    except:
        continue
    # reset the regression models
    #del regP, regS

ax[0].legend(fontsize=12, loc=3)
ax[1].legend(fontsize=12, loc=3)
ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
ax[0].set_xlabel('Distance along cable (km)')
ax[1].set_xlabel('Distance along cable (km)')
ax[0].set_ylabel('Site terms P (in log10)')
ax[1].set_ylabel('Site terms S (in log10)')
#ax[1].invert_xaxis()

plt.figure(fig.number)
plt.savefig(regression_results_dir + '/compare_site_terms.png', bbox_inches='tight')
plt.savefig(regression_results_dir + '/compare_site_terms.pdf', bbox_inches='tight')
# %% Compare the regression parameters and site terms
# compare with Yang 2022 for Ridgecrest 
data = np.load('/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/site_amplification_Yang2022.npz')
offset = -(data['offset'] - data['offset'][0])
# compare with Tomography results from Ettore
filename = "/kuafu/ebiondi/research/projects/LongValley/models/Profiles/DD_tomoCVM.npz"
with np.load(filename, allow_pickle=True) as fid:
    VpN = fid["VpN"]
    VpS = fid["VpS"]
    VsN = fid["VsN"]
    VsS = fid["VsS"]

    dzProf = fid["dz"][0]
    ozProf = fid["oz"][0]
nChN = VpN.shape[0]
nChS = VpS.shape[0]
nzProf = VpN.shape[1]

depth_z = ozProf+np.arange(nzProf)*dzProf

z_range = 1

VpN_2km = np.nanmean(VpN[:, depth_z<=z_range], axis=1)
VpS_2km = np.nanmean(VpS[:, depth_z<=z_range], axis=1)
VsN_2km = np.nanmean(VsN[:, depth_z<=z_range], axis=1)
VsS_2km = np.nanmean(VsS[:, depth_z<=z_range], axis=1)

VpN_2km = (VpN_2km - np.mean(VpN_2km))*5+1.5
VpS_2km = (VpS_2km - np.mean(VpS_2km))*5+1.5
VsN_2km = (VsN_2km - np.mean(VsN_2km))*5+1.5
VsS_2km = (VsS_2km - np.mean(VsS_2km))*5+1.5

# fig, ax = plt.subplots(figsize=(20,16))
# im = ax.imshow(VpN.T, cmap=plt.get_cmap("jet_r"),
#                extent=[0, nCh, ozProf+(nzProf-1)*dzProf, ozProf], vmin=np.nanmin(VpN),
#                vmax=np.nanmax(VpN), aspect=100.0)
# ax.set_xlabel("Channel number")
# ax.set_ylabel("z [km]")
# # ax.set_xlim(min_x,max_x)
# ax.grid()
# axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
#                    bbox_to_anchor = (1.02, 0.0, 1, 1), bbox_transform = ax.transAxes,
#                    borderpad = 0)
# cbar = fig.colorbar(im, cax=axins, orientation='vertical')
# cbar.set_label('Vp [km/s]')

#%%
fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
# DataFrame to store parameters for all models
P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'], 
index = np.arange(len(combined_channel_number_list)))
S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
index = np.arange(len(combined_channel_number_list)))

for i_model, combined_channel_number in enumerate(combined_channel_number_list):

    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")

    peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)
    combined_channel_id = np.sort(peak_amplitude_df.combined_channel_id.unique())
    
# Compare all the site terms
    site_term_P = regP.params[:-2]
    site_term_S = regS.params[:-2]

    if combined_channel_number == 1:
        ax[0].plot(combined_channel_id*channel_space, 10**site_term_P, label=f'Individual site terms, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(combined_channel_id*channel_space, 10**site_term_S, label=f'Individual site terms, Cond.# {regS.condition_number:.2f}')
    elif combined_channel_number == -1:
        ax[0].hlines(10**site_term_P, xmin=DAS_index.min()*channel_space, xmax=DAS_index.max()*channel_space, color='gray', label=f'Same site terms')#, Cond.# {regP.condition_number:.2f}')
        ax[1].hlines(10**site_term_S, xmin=DAS_index.min()*channel_space, xmax=DAS_index.max()*channel_space, color='gray', label=f'Same site terms')  #Cond.# {regS.condition_number:.2f}')
    else:
        ax[0].plot(combined_channel_id * combined_channel_number * channel_space, 10**np.array(site_term_P), '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(combined_channel_id * combined_channel_number * channel_space, 10**site_term_S, '-', label=f'{combined_channel_number} channels')#, Cond.# {regS.condition_number:.2f}')

if region_label == 'ridgecrest':
    ax[1].plot((np.arange(len(offset))+123)*channel_space, 10**data['log10S'], '-k', linewidth=3, label='S-wave site amplification\n from Yang et al., (2022)')
    # ax[1].fill_between((np.arange(len(offset))+123)*channel_space, 10**(data['log10S']-data['log10Serr']), 10**(data['log10S']+ data['log10Serr']),
    #                    color='r', alpha=0.2)
    ax[1].plot((np.arange(len(offset))+123)*channel_space, data['vs30']/1000*10-1.5, '-r', linewidth=3, label='Vs30 from Yang et al., (2022)')
                
elif (region_label == 'mammothS'):
    ax[0].plot(np.arange(nChS) * channel_space, VpS_2km, '-k', linewidth=3, label=f'P-wave velocity\n ({z_range}km) from Ettore')
    ax[1].plot(np.arange(nChS) * channel_space, VsS_2km, '-k', linewidth=3, label=f'S-wave velocity\n ({z_range}km) from Ettore')

elif (region_label == 'mammothN'):
    ax[0].plot(np.arange(nChN) * channel_space, VpN_2km, '-k', linewidth=3, label=f'P-wave velocity\n ({z_range}km) from Ettore')
    ax[1].plot(np.arange(nChN) * channel_space, VsN_2km, '-k', linewidth=3, label=f'S-wave velocity\n ({z_range}km) from Ettore')


ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12, loc=1)
ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
ax[0].set_xlabel('Distance along cable (km)')
ax[1].set_xlabel('Distance along cable (km)')
ax[0].set_ylabel('Site terms P')
ax[1].set_ylabel('Site terms S')
plt.figure(fig.number)
plt.savefig(regression_results_dir + '/compare_site_terms.png', bbox_inches='tight')



# %%
# ==============================  Looking into the multiple array case ========================================
# Specify the file names
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'

weighted = '_weighted'
# directory to store the fitted results:
regression_results_dir = results_output_dir + f'/regression_results_smf{weighted}_100_channel_at_least'
# %% 
# Combine site terms
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
    

# # select the site terms in different regions
# site_term_df_RC = site_term_df[site_term_df.region_site.str.contains('ridgecrest')]
# %% 
# Compare the regression parameters and site terms
show_calibrated = True#False # show the site terms with or without secondary calibration
fig, ax = plt.subplots(4, 2, figsize=(16, 17))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
title_plot = ['Ridgecrest P', 'Ridgecrest S', 'Long Valley North P', 'Long Valley North S', 'Long Valley South P', 'Long Valley South S']
region_labels = ['ridgecrest', 'mammothN', 'mammothS']
channel_spacing = [8e-3, 10e-3, 10e-3]
label_lists = ['10 channels','20 channels','50 channels','100 channels','constant']
# specify the ylim of figures...
region_y_minP, region_y_maxP = [0, -1, -1], [1.5, 1.5, 1.5]
region_y_minS, region_y_maxS = [0.2, -1.2, -1, -1.5], [1, 1.2, 1, 0]
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    site_term_df0 = pd.read_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')
    site_term_df0 = site_term_df0.sort_values(by=['channel_id'])
    second_calibration =pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_{combined_channel_number}chan.csv')
    # to obtain the site term calibration
    site_term_df_calibrated = pd.merge(site_term_df0, second_calibration, on=['region_site', 'channel_id'])

    if show_calibrated:
        site_term_df = site_term_df_calibrated
        site_term_df.site_term_P = site_term_df.site_term_P + site_term_df.diff_peak_P
        site_term_df.site_term_S = site_term_df.site_term_S + site_term_df.diff_peak_S
    else:
        site_term_df = site_term_df0
        
    for i_region, region_label in enumerate(region_labels):
        site_term_df_now = site_term_df[site_term_df.region_site.str.contains(region_label)]
        #site_term_df0_now = site_term_df0[site_term_df0.region_site.str.contains(region_label)]
        site_term_df_now = site_term_df_now.sort_values(by='channel_id')
        
        gca = ax[i_region, 0]
        if i_region != 2: 
            gca.plot(site_term_df_now.channel_id*channel_spacing[i_region], site_term_df_now.site_term_P)
            #gca.plot(site_term_df0_now.channel_id*channel_spacing[i_region], site_term_df0_now.site_term_P)
        else:
            gca.plot(site_term_df_now.channel_id*channel_spacing[i_region], site_term_df_now.site_term_P, label=label_lists[i_model])
        gca.set_title(title_plot[i_region * 2])
        gca.set_ylim(region_y_minP[i_region], region_y_maxP[i_region])
        #gca.grid()

        gca = ax[i_region, 1]
        gca.plot(site_term_df_now.channel_id*channel_spacing[i_region], site_term_df_now.site_term_S)
        gca.set_title(title_plot[i_region * 2 + 1])
        gca.set_ylim(region_y_minS[i_region], region_y_maxS[i_region])
        
        #gca.grid()
        #gca.sharey(ax[i_region, 0])

gca = ax[3, 0]
fig.delaxes(gca)

# add Sanriku
sanriku_regression_results_dir = f'/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/regression_results_smf{weighted}_all_coefficients_drop_4130_100_channel_at_least'
combined_channel_number_list=[10, 20, 50, 100]

gca = ax[3, 1]
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    site_term_df = pd.read_csv(sanriku_regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')
    if show_calibrated:
        gca.plot(site_term_df.channel_id*5/1e3, site_term_df.site_term_S + site_term_df.diff_peak_S, '-')
    else:
        gca.plot(site_term_df.channel_id*5/1e3, site_term_df.site_term_S, '-')
#gca.grid()
gca.set_title('Sanriku S')
gca.set_ylim(region_y_minS[3], region_y_maxS[3])

ax[i_region, 0].legend(bbox_to_anchor=(0.8, -0.4), loc=1)

for ii in range(3):
    ax[ii, 0].set_ylabel('Site term (in log10)')
ax[2, 0].set_xlabel('Distance (km)')
ax[3, 1].set_xlabel('Distance (km)')


letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for i_ax, gca in enumerate(ax.flatten()):
    if i_ax !=6:
        gca.annotate(f'({letter_list[k]})', xy=(-0.1, 1.1), xycoords=gca.transAxes, fontsize=20)
        k+=1

if show_calibrated:
    plt.savefig(results_output_dir + f'/compare_site_terms{weighted}.pdf', bbox_inches='tight')
    plt.savefig(results_output_dir + f'/compare_site_terms{weighted}.png', bbox_inches='tight')
else:
    plt.savefig(results_output_dir + f'/compare_site_terms{weighted}_uncalibrated.pdf', bbox_inches='tight')
    plt.savefig(results_output_dir + f'/compare_site_terms{weighted}_uncalibrated.png', bbox_inches='tight')    
# %%
# show channel numbers
fig, ax = plt.subplots(2, 1, figsize=(16, 8), squeeze=False, sharex=True)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
title_plot = ['Ridgecrest P', 'Ridgecrest S', 'Long Valley North P', 'Long Valley North S', 'Long Valley South P', 'Long Valley South S']
region_labels = ['ridgecrest', 'mammothN', 'mammothS']
channel_spacing = [8e-3, 10e-3, 10e-3]
label_lists = ['10 channels','20 channels','50 channels','100 channels','constant']
combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model


region_labels = ['ridgecrest']
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')

    for i_region, region_label in enumerate(region_labels):
        site_term_df_now = site_term_df[site_term_df.region_site.str.contains(region_label)]
        site_term_df_now = site_term_df_now.sort_values(by='channel_id')
        
        gca = ax[0, i_region]
        if i_region != 2: 
            gca.plot(site_term_df_now.channel_id, site_term_df_now.site_term_P)
        else:
            gca.plot(site_term_df_now.channel_id, site_term_df_now.site_term_P, label=label_lists[i_model])
        gca.set_title(title_plot[i_region * 2])
        #gca.grid()

        gca = ax[1, i_region]
        gca.plot(site_term_df_now.channel_id, site_term_df_now.site_term_S)
        gca.set_title(title_plot[i_region * 2 + 1])
        #gca.grid()
        #gca.sharey(ax[i_region, 0])

gca.set_xlim(550, 650)

# %%
# list mean site terms
site_term_df0 = pd.read_csv(regression_results_dir + f'/site_terms_100chan.csv')
site_term_df0 = site_term_df0.sort_values(by=['channel_id'])
second_calibration =pd.read_csv(regression_results_dir + f'/secondary_site_terms_calibration_100chan.csv')
# to obtain the site term calibration
site_term_df_calibrated = pd.merge(site_term_df0, second_calibration, on=['region_site', 'channel_id'])

if show_calibrated:
    site_term_df = site_term_df_calibrated
    site_term_df.site_term_P = site_term_df.site_term_P + site_term_df.diff_peak_P
    site_term_df.site_term_S = site_term_df.site_term_S + site_term_df.diff_peak_S
else:
    site_term_df = site_term_df0
    
for i_region, region_label in enumerate(region_labels):
    site_term_df_now = site_term_df[site_term_df.region_site.str.contains(region_label)]

    mean_site_term_P = np.nanmean(site_term_df_now.site_term_P)
    mean_site_term_S = np.nanmean(site_term_df_now.site_term_S)
    print(region_label)
    print(f'site term mean P {mean_site_term_P}')
    print(f'site term mean S {mean_site_term_S}')


# add Sanriku
sanriku_regression_results_dir = f'/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/regression_results_smf{weighted}_all_coefficients_drop_4130_100_channel_at_least'
combined_channel_number_list=[10, 20, 50, 100]

for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    site_term_df = pd.read_csv(sanriku_regression_results_dir + f'/site_terms_100chan.csv')
    mean_site_term_S = np.nanmean(site_term_df.site_term_S)
print('Sanriku')
print(f'site term mean S {mean_site_term_S}')
# %%

