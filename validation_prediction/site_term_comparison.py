#%% import modules
import pandas as pd
#from sep_util import read_file
import pandas as pd

import sys
sys.path.append('../')

import matplotlib
import matplotlib.pyplot as plt

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
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42 # Turn off text conversion to outlines
}
matplotlib.rcParams.update(params)

from utility.plotting import add_annotate

#%%
# Compare the site terms from multi arrays and single arrays
# multiple arrays
results_output_dir_multi = '../iter_results'
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
results_output_dir = '../iter_results_Ridgecrest'
region_dx = 8
region_key = 'ridgecrest'
region_text = 'Ridgecrest'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# Long Valley N
results_output_dir = '../iter_results_LongValley_N'
region_dx = 10
region_key = 'mammothN'
region_text = 'Long Valley (N)'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# Long Valley S
results_output_dir = '../iter_results_LongValley_S'
region_dx = 10
region_key = 'mammothS'
region_text = 'Long Valley (S)'
results_output_dir_list.append(results_output_dir)
region_key_list.append(region_key)
region_text_list.append(region_text)
region_dx_list.append(region_dx)

# Show comparison
# regression_results_dir_multi = results_output_dir_multi + f'/iter_regression_results_smf_weighted_{min_channel}_channel_at_least'
site_term_df_multi = pd.read_csv(results_output_dir_multi + '/site_terms_iter.csv')

fig, ax = plt.subplots(3, 2, figsize=(16, 12), )
for i_region, results_output_dir_single in enumerate(results_output_dir_list):
    region_key, region_dx = region_list[i_region], dx_list[i_region]

    site_term_df_single = pd.read_csv(results_output_dir_single + '/site_terms_iter.csv')

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

plt.savefig('../data_figures/site_terms_compare.png', bbox_inches='tight')

# %%
