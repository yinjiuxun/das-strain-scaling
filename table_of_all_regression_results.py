#%% import modules
from cmath import e
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
#%%
results_output_dir_list = ['/home/yinjx/kuafu/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate',
                      '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North',
                      '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South',
                      '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate',
                      '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM']
region_list = ['Ridgecrest', 'Long-Valley N', 'Long-Valley S', 'Sanriku', 'combined']
regression_dir_list = ['regression_results_smf_100_channel_at_least', 'regression_results_smf_weighted_100_channel_at_least'] # 'regression_results_smf_M4'
output_label = ['unweighted', 'weighted']
nearby_channel_numbers = [-1, 10, 20, 50, 100]




                                   
all_results_pd_weighted = pd.DataFrame(columns={'region', 'site-term smoothing', 
                                       'mag coef. (P)', 'dist coef. (P)', 'mag coef. uncertainty (P)', 'dist coef. uncertainty (P)',
                                       'mag coef. (S)', 'dist coef. (S)', 'mag coef. uncertainty (S)', 'dist coef. uncertainty (S)'})

#%%
def uncertainty_from_covariance(cov, key):
    return np.sqrt(cov.loc[key, key])


i_row = 0 # used to assign values

ii_weight = 1 # 0 for unweight, 1 for weighted
for ii_weight in [0, 1]:
    all_results_pd = pd.DataFrame(columns=['region', 'site-term smoothing', 
                                       'mag coef. (P)', 'dist coef. (P)', 'mag coef. uncertainty (P)', 'dist coef. uncertainty (P)',
                                       'mag coef. (S)', 'dist coef. (S)', 'mag coef. uncertainty (S)', 'dist coef. uncertainty (S)'])
                                       
    for ii_region, results_output_dir in enumerate(results_output_dir_list):
    # ii_region = 0 
    # results_output_dir = results_output_dir_list[ii_region]
        for nearby_channel_number in nearby_channel_numbers:
        #nearby_channel_number = nearby_channel_numbers[0]
        # for regression_dir in regression_dir_list:
            
            regression_dir = regression_dir_list[ii_weight]

            if region_list[ii_region] != 'Sanriku':
                regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
                regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
            elif nearby_channel_number == -1:
                i_row += 1
                continue
            else:
                if ii_weight==0:
                    regression_dir = 'regression_results_smf_all_coefficients_drop_4130_100_channel_at_least'
                else:
                    regression_dir = 'regression_results_smf_weighted_all_coefficients_drop_4130_100_channel_at_least'

                regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
            
            all_results_pd.at[i_row, 'region'] = region_list[ii_region]
            all_results_pd.at[i_row, 'site-term smoothing'] = nearby_channel_number
            if region_list[ii_region] == 'Sanriku':
                all_results_pd.at[i_row, 'mag coef. (P)'] = np.nan
                all_results_pd.at[i_row, 'dist coef. (P)'] = np.nan
                all_results_pd.at[i_row, 'mag coef. uncertainty (P)'] = np.nan
                all_results_pd.at[i_row, 'dist coef. uncertainty (P)'] = np.nan
            else:
                all_results_pd.at[i_row, 'mag coef. (P)'] = regP.params['magnitude']
                all_results_pd.at[i_row, 'dist coef. (P)'] = regP.params['np.log10(distance_in_km)']
                all_results_pd.at[i_row, 'mag coef. uncertainty (P)'] = uncertainty_from_covariance(regP.cov_params(), 'magnitude')
                all_results_pd.at[i_row, 'dist coef. uncertainty (P)'] = uncertainty_from_covariance(regP.cov_params(), 'np.log10(distance_in_km)')

            all_results_pd.at[i_row, 'mag coef. (S)'] = regS.params['magnitude']
            all_results_pd.at[i_row, 'dist coef. (S)'] = regS.params['np.log10(distance_in_km)']
            all_results_pd.at[i_row, 'mag coef. uncertainty (S)'] = uncertainty_from_covariance(regS.cov_params(), 'magnitude')
            all_results_pd.at[i_row, 'dist coef. uncertainty (S)'] = uncertainty_from_covariance(regS.cov_params(), 'np.log10(distance_in_km)')

            i_row += 1

    for ii_column in range(2, 10):
        all_results_pd[all_results_pd.columns[ii_column]] = pd.to_numeric(all_results_pd[all_results_pd.columns[ii_column]])

    # 
    all_results_pd.iloc[:, 2:] = all_results_pd.iloc[:, 2:].astype('float')
    all_results_pd.to_csv(f'/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/all_coefficients_{output_label[ii_weight]}.csv', 
                        index=False, float_format='%.4f')
# %%
all_results_pd_unweighted = pd.read_csv('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/all_coefficients_unweighted.csv')
all_results_pd_weighted = pd.read_csv('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/all_coefficients_weighted.csv')
all_results_pd_unweighted['weight'] = 'unweighted'
all_results_pd_weighted['weight'] = 'weighted'

all_results_pd = pd.concat([all_results_pd_unweighted, all_results_pd_weighted], axis=0)
# %%
# Only plot ST-100 with error bar
import seaborn as sns


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale=1.2)


fig, ax = plt.subplots(2, 2, figsize=(12, 9))
def plot_coefficients_seaborn(x_key, y_key, all_results_pd, ax, xlim=None, ylim=None):
    # g = sns.pointplot(x=x_key, y=y_key, hue='site-term smoothing',
    #             capsize=.2, palette="crest", markers="o", linestyles='', scale=1.5, linewidth=2, edgecolor='k',
    #             kind=x_key, data=all_results_pd[all_results_pd.weight=='unweighted'], ax=ax, legend = False)

    g = sns.pointplot(x="region", y=y_key, hue='site-term smoothing', ci=y_key[:-3]+'uncertainty '+y_key[-3:],
                capsize=.2, palette="crest", markers="d", linestyles='', scale=2, 
                kind="point", data=all_results_pd[all_results_pd.weight=='weighted'], ax=ax, legend = False)
    
    # g = sns.scatterplot(data=all_results_pd[all_results_pd.weight=='unweighted'],
    #                     x=x_key, y=y_key, s=100, hue='site-term smoothing',
    #                     palette="Blues", markers="o",linewidth=1, edgecolor='k',
    #                     ax=ax, legend = True)

    # g = sns.scatterplot(data=all_results_pd[all_results_pd.weight=='weighted'], 
    #                     x="region", y=y_key, s=100, hue='site-term smoothing',
    #                     palette="Reds", markers="d",linewidth=1, edgecolor='k',
    #                     ax=ax, legend = True)
    
    ax.set_xticklabels(['RC', 'LV-N', 'LV-S', 'Sanriku', 'combined'])
    ax.set_yticks(np.arange(-10, 10, 0.2))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid()

    return g


plot_coefficients_seaborn(x_key='region', y_key='mag coef. (P)', 
            all_results_pd = all_results_pd[all_results_pd['site-term smoothing']==100], ax=ax[0, 0], ylim=[0.0, 1])

plot_coefficients_seaborn(x_key='region', y_key='mag coef. (S)', 
            all_results_pd = all_results_pd[all_results_pd['site-term smoothing']==100], ax=ax[0, 1], ylim=[0.0, 1])

plot_coefficients_seaborn(x_key='region', y_key='dist coef. (P)', 
            all_results_pd = all_results_pd[all_results_pd['site-term smoothing']==100], ax=ax[1, 0], ylim=[-2, -0.5])

plot_coefficients_seaborn(x_key='region', y_key='dist coef. (S)', 
            all_results_pd = all_results_pd[all_results_pd['site-term smoothing']==100], ax=ax[1, 1], ylim=[-2, -0.5])

# Adding the Barbour et al. (2021) results
# TODO: label the line with the values
barbour_2021_coefficents = [0.92, -1.45]
PGA_coefficients = [0.388142, -1.630351]
ax[0, 0].hlines(y=PGA_coefficients[0], xmin=0, xmax=4, linestyle='--', color='k', linewidth=2)
ax[0, 1].hlines(y=PGA_coefficients[0], xmin=0, xmax=4, linestyle='--', color='k', linewidth=2)
ax[1, 0].hlines(y=PGA_coefficients[1], xmin=0, xmax=4, linestyle='--', color='k', linewidth=2)
ax[1, 1].hlines(y=PGA_coefficients[1], xmin=0, xmax=4, linestyle='--', color='k', linewidth=2, label='PGA from NGA-West2')

ax[0, 0].get_legend().remove()
ax[0, 1].get_legend().remove()
ax[1, 0].get_legend().remove()

my_label = ['Regression', 'PGA from NGA-West2']#['uniform ST', 'ST-10', 'ST-20', 'ST-50', 'ST-100', 'Barbour et al. (2021)'] 
L = ax[1, 1].legend(loc='center left', bbox_to_anchor=(-1.4, 2.5), ncol=6, title='regression coefficients')
for i_L in range(len(L.get_texts())):
    L.get_texts()[i_L].set_text(my_label[i_L])


letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for gca in ax.flatten():
    if gca is not None:
        gca.annotate(f'({letter_list[k]})', xy=(-0.2, 1.0), xycoords=gca.transAxes, fontsize=18)
        k += 1

# TODO: add error bars
# TODO: edit the legend box

plt.savefig('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/coefficients_comparison.png', bbox_inches='tight', dpi=200)
plt.savefig('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/coefficients_comparison.pdf', bbox_inches='tight')

# %%
# plot all without errorbars
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale=1.2)


fig, ax = plt.subplots(2, 2, figsize=(12, 9))
def plot_coefficients_seaborn(x_key, y_key, all_results_pd, ax, xlim=None, ylim=None):
    g = sns.pointplot(x=x_key, y=y_key, hue='site-term smoothing',
                capsize=.2, palette="flare", markers="o", linestyles='', scale=1.5, linewidth=2, edgecolor='k',
                kind=x_key, data=all_results_pd[all_results_pd.weight=='unweighted'], ax=ax, legend = False)

    g = sns.pointplot(x=x_key, y=y_key, hue='site-term smoothing',
                capsize=.2, palette="crest", markers="d", linestyles='', scale=1.5, 
                kind="point", data=all_results_pd[all_results_pd.weight=='weighted'], ax=ax, legend = False)
    
    # g = sns.scatterplot(data=all_results_pd[all_results_pd.weight=='unweighted'],
    #                     x=x_key, y=y_key, s=100, hue='site-term smoothing',
    #                     palette="Blues", markers="o",linewidth=1, edgecolor='k',
    #                     ax=ax, legend = True)

    # g = sns.scatterplot(data=all_results_pd[all_results_pd.weight=='weighted'], 
    #                     x="region", y=y_key, s=100, hue='site-term smoothing',
    #                     palette="Reds", markers="d",linewidth=1, edgecolor='k',
    #                     ax=ax, legend = True)
    
    ax.set_xticklabels(['RC', 'LV-N', 'LV-S', 'Sanriku', 'combined'])
    # ax.set_yticks(np.arange(-10, 10, 0.2))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid()

    return g



plot_coefficients_seaborn(x_key='region', y_key='mag coef. (P)', all_results_pd = all_results_pd, ax=ax[0, 0], ylim=[0.0, 1])

plot_coefficients_seaborn(x_key='region', y_key='mag coef. (S)', all_results_pd = all_results_pd, ax=ax[0, 1], ylim=[0.0, 1])

plot_coefficients_seaborn(x_key='region', y_key='dist coef. (P)', all_results_pd = all_results_pd, ax=ax[1, 0], ylim=[-2, -0.5])

plot_coefficients_seaborn(x_key='region', y_key='dist coef. (S)', all_results_pd = all_results_pd, ax=ax[1, 1], ylim=[-2, -0.5])

# Adding the Barbour et al. (2021) results
# TODO: label the line with the values
barbour_2021_coefficents = [0.92, -1.45]
PGA_coefficients_OLS = [0.583631, -1.793554]
PGA_coefficients_WLS = [0.388142, -1.630351]
ax[0, 0].hlines(y=PGA_coefficients_OLS[0], xmin=0, xmax=4, linestyle='--', color='r', linewidth=2)
ax[0, 1].hlines(y=PGA_coefficients_OLS[0], xmin=0, xmax=4, linestyle='--', color='r', linewidth=2)
ax[1, 0].hlines(y=PGA_coefficients_OLS[1], xmin=0, xmax=4, linestyle='--', color='r', linewidth=2)
ax[1, 1].hlines(y=PGA_coefficients_OLS[1], xmin=0, xmax=4, linestyle='--', color='r', linewidth=2, label='PGA from NGA-West2 (OLS)')


ax[0, 0].hlines(y=PGA_coefficients_WLS[0], xmin=0, xmax=4, linestyle='--', color='b', linewidth=2)
ax[0, 1].hlines(y=PGA_coefficients_WLS[0], xmin=0, xmax=4, linestyle='--', color='b', linewidth=2)
ax[1, 0].hlines(y=PGA_coefficients_WLS[1], xmin=0, xmax=4, linestyle='--', color='b', linewidth=2)
ax[1, 1].hlines(y=PGA_coefficients_WLS[1], xmin=0, xmax=4, linestyle='--', color='b', linewidth=2, label='PGA from NGA-West2 (WLS)')


ax[0, 0].get_legend().remove()
ax[0, 1].get_legend().remove()
ax[1, 0].get_legend().remove()

my_label = ['all (OLS)', '10 (OLS)', '20 (OLS)', '50 (OLS)', '100 (OLS)', 'all (WLS)', 
            '10 (WLS)', '20 (WLS)', '50 (WLS)', '100 (WLS)',  'PGA from NGA-West2 (OLS)', 'PGA from NGA-West2 (WLS)'] #
L = ax[1, 1].legend(loc='center left', bbox_to_anchor=(-1.4, 2.5), ncol=6, title='regression coefficients')
for i_L in range(len(L.get_texts())):
    L.get_texts()[i_L].set_text(my_label[i_L])


letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for gca in ax.flatten():
    if gca is not None:
        gca.annotate(f'({letter_list[k]})', xy=(-0.2, 1.0), xycoords=gca.transAxes, fontsize=18)
        k += 1


# TODO: add error bars
# TODO: edit the legend box
plt.savefig('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/coefficients_comparison_all.png', bbox_inches='tight', dpi=200)
plt.savefig('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RMS/coefficients_comparison_all.pdf', bbox_inches='tight')
# %%
