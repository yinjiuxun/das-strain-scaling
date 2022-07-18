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
fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    try:
        site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{combined_channel_number}chan.csv')

        ax[0].plot(site_term_df.channel_id*channel_space, site_term_df.site_term_P, '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')
        ax[1].plot(site_term_df.channel_id*channel_space, site_term_df.site_term_S, '-', label=f'{combined_channel_number} channels')#, Cond.# {regS.condition_number:.2f}')
    except:
        continue
    # reset the regression models
    #del regP, regS

ax[0].legend(fontsize=12)
ax[1].legend(fontsize=12)
ax[0].xaxis.set_tick_params(which='both',labelbottom=True)
ax[0].set_xlabel('Distance along cable (km)')
ax[1].set_xlabel('Distance along cable (km)')
ax[0].set_ylabel('Site terms P (in log10)')
ax[1].set_ylabel('Site terms S (in log10)')
#ax[1].invert_xaxis()

plt.figure(fig.number)
plt.savefig(regression_results_dir + '/compare_site_terms.png', bbox_inches='tight')

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

# directory to store the fitted results
regression_results_dir = results_output_dir + '/regression_results_smf_weighted'

# %% Compare the regression parameters and site terms
fig, ax = plt.subplots(3, 2, figsize=(20, 22))#, sharex=True, sharey=True)
ax = ax.flatten()

title_plot = ['Ridgecrest P', 'Ridgecrest S', 'Long Valley North P', 'Long Valley North S', 'Long Valley South P', 'Long Valley South S']

combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
for i_model, combined_channel_number in enumerate(combined_channel_number_list):
    peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{combined_channel_number}.csv')

    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")

    # peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)
    # combined_channel_id = np.sort(peak_amplitude_df.combined_channel_id.unique())
    # peak_amplitude_df = add_event_label(peak_amplitude_df)

    
# extract the site terms for different arrays
    combined_channel_ridgecrest, site_term_P_ridgecrest = process_region_param_keys(regP, 'ridgecrest')
    combined_channel_ridgecrest, site_term_S_ridgecrest = process_region_param_keys(regS, 'ridgecrest')

    combined_channel_north, site_term_P_north = process_region_param_keys(regP, 'mammothN')
    combined_channel_north, site_term_S_north = process_region_param_keys(regS, 'mammothN')

    combined_channel_south, site_term_P_south = process_region_param_keys(regP, 'mammothS')
    combined_channel_south, site_term_S_south = process_region_param_keys(regS, 'mammothS')

    combined_channel_list_plot = [combined_channel_ridgecrest, combined_channel_ridgecrest, 
                                  combined_channel_north, combined_channel_north,
                                  combined_channel_south, combined_channel_south]
    site_term_list_plot = [site_term_P_ridgecrest, site_term_S_ridgecrest,
                           site_term_P_north, site_term_S_north,
                           site_term_P_south, site_term_S_south]

    for ii_region in range(0, 6):
        gca = ax[ii_region]

        if combined_channel_number == -1:
            gca.hlines(site_term_list_plot[ii_region], xmin=0, xmax=len(combined_channel_list_plot[ii_region])*combined_channel_number, 
            color='k', label=f'Same site terms')#, Cond.# {regP.condition_number:.2f}')

        else:
            gca.plot(combined_channel_list_plot[ii_region] * combined_channel_number, site_term_list_plot[ii_region],
             '-', label=f'{combined_channel_number} channels')#, Cond.# {regP.condition_number:.2f}')

        gca.set_xlabel('Channel number')
        gca.set_ylabel('Site terms')
        gca.set_title(title_plot[ii_region])

ax[0].legend(fontsize=12)
gca.xaxis.set_tick_params(which='both',labelbottom=True)


# Adding annotation
letter_list = [str(chr(k+97)) for k in range(0, 20)]
k=0
for gca in ax:
    if gca is not None:
        gca.annotate(f'({letter_list[k]})', xy=(-0.1, 1.0), xycoords=gca.transAxes)
        k += 1


#ax[1].invert_xaxis()

plt.figure(fig.number)
plt.savefig(regression_results_dir + '/compare_site_terms_all.png', bbox_inches='tight')

# plt.figure(fig2.number)
# plt.savefig(regression_results_dir + '/compare_site_terms_south.png', bbox_inches='tight')
# %%
# #Test parameters from strain meter (Not right)
# fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)
# # combined_channel_number_list = [10, 20, 50, 100, -1] # -1 means the constant model
# combined_channel_number_list = [100]
# # DataFrame to store parameters for all models
# P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'], 
# index = np.arange(len(combined_channel_number_list)))
# S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
# index = np.arange(len(combined_channel_number_list)))

# for i_model, combined_channel_number in enumerate(combined_channel_number_list):

#     regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{combined_channel_number}chan.pickle")
#     regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{combined_channel_number}chan.pickle")
#     # apply parameters from strain meter
#     regP.params['magnitude'] = 0.92
#     regP.params['np.log10(distance_in_km)'] = -1.45

#     regS.params['magnitude'] = 0.92
#     regS.params['np.log10(distance_in_km)'] = -1.45

#     peak_amplitude_df = combined_channels(DAS_index, peak_amplitude_df, combined_channel_number)
#     combined_channel_id = np.sort(peak_amplitude_df.combined_channel_id.unique())
#     peak_amplitude_df = add_event_label(peak_amplitude_df)

#     y_P_predict = regP.predict(peak_amplitude_df)
#     y_S_predict = regS.predict(peak_amplitude_df)

#     temp_peaks = np.array([np.array(peak_amplitude_df.peak_P),
#               np.array(peak_amplitude_df.peak_S), 
#               np.array(10**y_P_predict), 
#               np.array(10**y_S_predict)]).T
#     peak_comparison_df = pd.DataFrame(data=temp_peaks,
#                                   columns=['peak_P', 'peak_S', 'peak_P_predict', 'peak_S_predict'])
    
#     g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 1000], phase='P')
#     g.savefig(regression_results_dir + f'/P_validate_predicted_combined_site_terms_{combined_channel_number}chan_strainmeter_seaborn.png')

#     g = plot_prediction_vs_measure_seaborn(peak_comparison_df, [0.01, 1000], phase='S')
#     g.savefig(regression_results_dir + f'/S_validate_predicted_combined_site_terms_{combined_channel_number}chan_strainmeter_seaborn.png')
#     # plot_compare_prediction_vs_true_values(peak_amplitude_df, y_P_predict, y_S_predict, (-2.0, 2), 
#     # regression_results_dir + f'/validate_predicted__combined_site_terms_{combined_channel_number}chan.png')