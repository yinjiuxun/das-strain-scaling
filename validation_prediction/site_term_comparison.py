#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

import sys
sys.path.append('../')
from utility.general import mkdir

import matplotlib
import matplotlib.pyplot as plt

from utility.processing import filter_event, calculate_autocorrelation


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

region_y_minP, region_y_maxP = [-1, -1, -1], [1.5, 1.5, 1.5]
region_y_minS, region_y_maxS = [-1, -1, -1, -1.5], [1.5, 1.5, 1.5, 0]

results_output_dir_list = []
dx_list = []

#%%
#  # Set result directory
# multiple arrays
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
region_list = ['ridgecrest', 'mammothN', 'mammothS']
region_text = ['Ridgecrest', 'Long Valley (N)', 'Long Valley (S)']
color_list = ['#7DC0A6', '#ED926B', '#91A0C7', '#DA8EC0']
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

#%%
# Calculate the autocorrelation 
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax = ax.flatten()
for i_wavetype, wavetype in enumerate(site_term_list):
    for i_region in range(len(region_list)):
        region_key, region_dx = region_list[i_region], dx_list[i_region]
        temp = site_term_df[site_term_df.region == region_key]
        site_term_now = temp[f'site_term_{wavetype.upper()}']
        gca = ax[i_wavetype]

        site_term_acf = calculate_autocorrelation(site_term_now, symmetric=True)


        gca.plot(temp.channel_id*region_dx/1e3, site_term_acf, color=color_list[i_region], label=f'{region_text[i_region]}')
        gca.set_ylabel('CC.')
        gca.set_title(f'{wavetype.upper()} wave')
        gca.grid()
        if i_wavetype == 1:
            gca.set_xlabel('Distance to IU (km)')
    gca.legend(loc=1, fontsize=12)

ax = add_annotate(ax)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
mkdir(regression_results_dir + '/figures')
plt.savefig(regression_results_dir + '/figures/site_terms_autocorrelation.png', bbox_inches='tight')

#%%
# Look into the histograms
site_term_range = (-1, 2)
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax = ax.flatten()
for i_wavetype, wavetype in enumerate(site_term_list):

    for i_region in range(len(region_list)):
        region_key, region_dx = region_list[i_region], dx_list[i_region]
        temp = site_term_df[site_term_df.region == region_key]

        site_term_now = temp[f'site_term_{wavetype.upper()}']

        gca = ax[i_wavetype]
        gca.hist(site_term_now, range=site_term_range, bins=50, label=f'{region_text[i_region]},\nmedian {np.median(site_term_now):.2f}', 
        alpha=0.5, density=True, color=color_list[i_region], edgecolor=None)
        gca.vlines(x=np.median(site_term_now), ymin=0, ymax=4, color=color_list[i_region])
    
    gca.legend(loc=2, fontsize=12)
    gca.set_title(f'{wavetype.upper()} wave')
    gca.set_xlabel('Site term in log10')

ax = add_annotate(ax)
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.savefig(regression_results_dir + '/figures/site_terms_histograms.png', bbox_inches='tight')

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
region_list = ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']
region_text = ['Ridgecrest', 'Long Valley (N)', 'Long Valley (S)']
site_term_list = ['P', 'S']
dx_list = [8, 10, 10, 5] # Ridgecrest, Mammoth N, Mammoth S, Sanriku
region_y_minP, region_y_maxP = [-1, -1, -1, -1], [1.5, 1.5, 1.5, 1.5]
region_y_minS, region_y_maxS = [-1, -1, -1, -1.5], [1.5, 1.5, 1.5, 1.5]

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




# %%
# Temporary use
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

    site_term_list = ['S']
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for i_wavetype, wavetype in enumerate(site_term_list):
        temp = site_term_df[site_term_df.region == region_key]

        
        peak_amplitude_df_temp = filter_event(peak_amplitude_df_fit, M_threshold, snr_threshold, min_channel)
        peak_amplitude_df_temp = peak_amplitude_df_temp[~peak_amplitude_df_temp[f'peak_{wavetype.upper()}'].isna()]
        
        gca = ax[0]
        try:
            gca.hist(peak_amplitude_df_temp.channel_id*region_dx/1e3, range=(0.5*region_dx/1e3, (peak_amplitude_df_temp.channel_id.max()+0.5)*region_dx/1e3), bins=int(peak_amplitude_df_temp.channel_id.max()), color='k')
        except:
            print(f"No {wavetype}, skip...")
        gca.set_ylabel('Number of \nmeasurements')
        
        gca = ax[1]
        try:
            #gca.plot(temp.channel_id*region_dx/1e3, temp[f'site_term_{wavetype.upper()}'], 'k.')
            gca.scatter(temp.channel_id*region_dx/1e3, temp[f'site_term_{wavetype.upper()}'],s=1, c='r')#temp[f'site_term_{wavetype.upper()}'], cmap='viridis')
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
    plt.savefig(regression_results_dir + '/figures/site_terms.pdf', bbox_inches='tight')
    plt.savefig(regression_results_dir + '/figures/site_terms.png', bbox_inches='tight')
# %%
