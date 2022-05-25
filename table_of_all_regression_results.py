#%% import modules
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
                      '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM']
region_list = ['Ridgecrest', 'Long-Valley N', 'Long-Valley S', 'combined']
regression_dir_list = ['regression_results_smf', 'regression_results_smf_weighted'] # 'regression_results_smf_M4'

nearby_channel_numbers = [-1, 10, 20, 50, 100]



all_results_pd = pd.DataFrame(columns=['region', 'site-term smoothing', 
                                       'mag coef. (P)', 'dist coef. (P)', 'mag coef. uncertainty (P)', 'dist coef. uncertainty (P)',
                                       'mag coef. (S)', 'dist coef. (S)', 'mag coef. uncertainty (S)', 'dist coef. uncertainty (S)'])
                                   
all_results_pd_weighted = pd.DataFrame(columns={'region', 'site-term smoothing', 
                                       'mag coef. (P)', 'dist coef. (P)', 'mag coef. uncertainty (P)', 'dist coef. uncertainty (P)',
                                       'mag coef. (S)', 'dist coef. (S)', 'mag coef. uncertainty (S)', 'dist coef. uncertainty (S)'})

#%%
def uncertainty_from_covariance(cov, key):
    return np.sqrt(cov.loc[key, key])


i_row = 0 # used to assign values

for ii_region, results_output_dir in enumerate(results_output_dir_list):
# ii_region = 0 
# results_output_dir = results_output_dir_list[ii_region]
    for nearby_channel_number in nearby_channel_numbers:
    #nearby_channel_number = nearby_channel_numbers[0]
    # for regression_dir in regression_dir_list:
        regression_dir = regression_dir_list[1]

        regP = sm.load(results_output_dir + '/' + regression_dir + f"/P_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
        regS = sm.load(results_output_dir + '/' + regression_dir + f"/S_regression_combined_site_terms_{nearby_channel_number}chan.pickle")
        
        all_results_pd.at[i_row, 'region'] = region_list[ii_region]
        all_results_pd.at[i_row, 'site-term smoothing'] = nearby_channel_number
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
all_results_pd.to_csv('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/all_coefficients_weighted.csv', 
                    index=False, float_format='%.4f')
# %%
all_results_pd_unweighted = pd.read_csv('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/all_coefficients.csv')
all_results_pd_weighted = pd.read_csv('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/all_coefficients_weighted.csv')
all_results_pd_unweighted['weight'] = 'unweighted'
all_results_pd_weighted['weight'] = 'weighted'

all_results_pd = pd.concat([all_results_pd_unweighted, all_results_pd_weighted], axis=0)
# %%
import seaborn as sns


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale=1.2)


fig, ax = plt.subplots(2, 2, figsize=(12, 9))
def plot_coefficients_seaborn(x_key, y_key, all_results_pd, ax, xlim=None, ylim=None):
    g = sns.pointplot(x=x_key, y=y_key, hue='site-term smoothing',
                capsize=.2, palette="Blues", markers="o", linestyles='', scale=1.5,
                kind=x_key, data=all_results_pd[all_results_pd.weight=='unweighted'], ax=ax, legend = False)


    g = sns.pointplot(x="region", y=y_key, hue='site-term smoothing',
                capsize=.2, palette="Reds", markers="d", linestyles='', scale=1.5,
                kind="point", data=all_results_pd[all_results_pd.weight=='weighted'], ax=ax, legend = False)
    
    ax.set_xticklabels(['RC', 'LV-N', 'LV-S', 'combined'])
    ax.set_yticks(np.arange(-10, 10, 0.2))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid()

    return g

plot_coefficients_seaborn(x_key='region', y_key='mag coef. (P)', all_results_pd = all_results_pd, ax=ax[0, 0], ylim=[0.3, 0.75])

plot_coefficients_seaborn(x_key='region', y_key='mag coef. (S)', all_results_pd = all_results_pd, ax=ax[0, 1], ylim=[0.3, 0.75])

plot_coefficients_seaborn(x_key='region', y_key='dist coef. (P)', all_results_pd = all_results_pd, ax=ax[1, 0], ylim=[-1.6, -0.7])

plot_coefficients_seaborn(x_key='region', y_key='dist coef. (S)', all_results_pd = all_results_pd, ax=ax[1, 1], ylim=[-1.6, -0.7])


ax[0, 0].get_legend().remove()
ax[0, 1].get_legend().remove()
ax[1, 0].get_legend().remove()
ax[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='site-term')

letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for gca in ax.flatten():
    if gca is not None:
        gca.annotate(f'({letter_list[k]})', xy=(-0.2, 1.0), xycoords=gca.transAxes, fontsize=18)
        k += 1


# TODO: add error bars
# TODO: edit the legend box

plt.savefig('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/coefficients_comparison.png', bbox_inches='tight', dpi=200)
# %%
