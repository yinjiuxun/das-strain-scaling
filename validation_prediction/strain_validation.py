#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import combined_channels, filter_event_first_order, get_comparison_df
from utility.regression import predict_strain
from utility.plotting import plot_prediction_vs_measure_seaborn

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

#%%
# some parameters
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [2, 10]
weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise

random_test = False # whether to plot the random test for transfered scaling

results_output_dir_list = []
regression_results_dir_list = []
peak_file_name_list = []
result_label_list = []
M_threshold_list = []
snr_threshold_list = []
vmax_list = []
region_text_list = []
plot_type_list = []

#%% # Set result directory
if not random_test:
    # ================== multiple arrays ================== 
    results_output_dir = '../iter_results'
    peak_file_name = '../data_files/peak_amplitude/peak_amplitude_multiple_arrays.csv'
    result_label = 'iter'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = 35000 
    region_text = 'California arrays'
    plot_type = 'histplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(results_output_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)
    plot_type_list.append(plot_type)

    # single arrays
    #  ================== Ridgecrest ================== 
    results_output_dir = '../iter_results_Ridgecrest'
    peak_file_name = '../data_files/peak_amplitude/peak_amplitude_Ridgecrest.csv'
    result_label = 'iter'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = 15000
    region_text = 'Ridgecrest'
    plot_type = 'histplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(results_output_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)
    plot_type_list.append(plot_type)

    #  ================== Long Valley N ================== 
    results_output_dir = '../iter_results_LongValley_N'
    peak_file_name = '../data_files/peak_amplitude/peak_amplitude_LongValley_N.csv'
    result_label = 'iter'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = 10000
    region_text = 'Long Valley North'
    plot_type = 'histplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(results_output_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)
    plot_type_list.append(plot_type)

    #  ================== Long Valley S ================== 
    results_output_dir = '../iter_results_LongValley_S'
    peak_file_name = '../data_files/peak_amplitude/peak_amplitude_LongValley_S.csv'
    result_label = 'iter'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = 10000
    region_text = 'Long Valley South'
    plot_type = 'histplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(results_output_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)
    plot_type_list.append(plot_type)

    #  ================== Sanriku fittd ================== 
    results_output_dir = '../iter_results_Sanriku'
    peak_file_name = '../data_files/peak_amplitude/peak_amplitude_Sanriku.csv'
    result_label = 'iter'
    snr_threshold = 5
    M_threshold = [2, 10]
    vmax = 1000 # for P and S
    region_text = 'Sanriku'
    plot_type = 'scatterplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(results_output_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)
    plot_type_list.append(plot_type)

else:
    #  ================== Sanriku transfered test ================== 
    N_event_fit_list = [5]
    N_test = 5
    for N_event_fit in N_event_fit_list:
        for i_test in range(N_test):

            results_output_dir = '../transfered_results'
            peak_file_name = '../data_files/peak_amplitude/peak_amplitude_Sanriku.csv'
            result_label = 'transfer'
            regression_results_dir = results_output_dir + f'/{N_event_fit}_fit_events_{i_test}th_test'
            snr_threshold = 5
            vmax = 50
            M_threshold = [0, 10]
            vmax = 1000 # for P and S
            region_text = 'Transfered scaling for Sanriku'
            plot_type = 'scatterplot'

            M_threshold_list.append(M_threshold)
            results_output_dir_list.append(results_output_dir)
            regression_results_dir_list.append(regression_results_dir)
            peak_file_name_list.append(peak_file_name)
            result_label_list.append(result_label)
            snr_threshold_list.append(snr_threshold)
            vmax_list.append(vmax)
            region_text_list.append(region_text) 
            plot_type_list.append(plot_type)

#%%
for ii in range(len(peak_file_name_list)):
    
    peak_file_name = peak_file_name_list[ii]
    regression_results_dir = regression_results_dir_list[ii]
    result_label = result_label_list[ii]
    snr_threshold = snr_threshold_list[ii]
    region_text = region_text_list[ii]
    results_output_dir = results_output_dir_list[ii]
    print(regression_results_dir)

    # load results
    peak_amplitude_df = pd.read_csv(peak_file_name)
    peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']

    peak_amplitude_df = filter_event_first_order(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)

    site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{result_label}.csv')

    if 'Sanriku' in results_output_dir: # some special processing for Sanriku data
        peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.QA == 'Yes']
        peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 4130].index)
        #peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 1580].index)

    try:
        regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{result_label}.pickle")
        peak_P_predicted, peak_amplitude_df_temp = predict_strain(peak_amplitude_df, regP, site_term_df, wavetype='P')
        peak_P_comparison_df = get_comparison_df(data=[peak_amplitude_df_temp.event_id, peak_amplitude_df_temp.peak_P, peak_P_predicted], 
                                            columns=['event_id', 'peak_P', 'peak_P_predict'])
    except:
        print('No P regression results, skip...')
        regP, peak_P_predicted, peak_amplitude_df_temp, peak_P_comparison_df = None, None, None, None


    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{result_label}.pickle")
        peak_S_predicted, peak_amplitude_df_temp = predict_strain(peak_amplitude_df, regS, site_term_df, wavetype='S')
        peak_S_comparison_df = get_comparison_df(data=[peak_amplitude_df_temp.event_id, peak_amplitude_df_temp.peak_S, peak_S_predicted], 
                                            columns=['event_id', 'peak_S', 'peak_S_predict'])
    except:
        print('No S regression results, skip...')
        regS, peak_S_predicted, peak_amplitude_df_temp, peak_S_comparison_df = None, None, None, None


    # plot figures of strain rate validation
    fig_dir = regression_results_dir + '/figures'
    mkdir(fig_dir)

    xy_lim, height, space = [1e-3, 1e2], 10, 0.3
    vmax = vmax_list[ii]

    if result_label == 'iter': 
        try:
            gP = plot_prediction_vs_measure_seaborn(peak_P_comparison_df, phase='P', bins=40, vmax=vmax, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gP.ax_joint.text(2e-3, 50, region_text + f', P wave\n{len(peak_P_comparison_df.dropna())} measurements')
            gP.savefig(fig_dir + f'/P_validate_strain_rate_{result_label}_seaborn.png')
        except:
            print('No P regression results, no figure, skip...')

        try:
            gS = plot_prediction_vs_measure_seaborn(peak_S_comparison_df, phase='S', bins=40, vmax=vmax, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gS.ax_joint.text(2e-3, 50, region_text + f', S wave\n{len(peak_S_comparison_df.dropna())} measurements')
            gS.savefig(fig_dir + f'/S_validate_strain_rate_{result_label}_seaborn.png')
        except:
            print('No S regression results, no figure, skip...')
        
    elif result_label == 'transfer':
        temp = np.load(regression_results_dir + '/transfer_event_list.npz')
        event_id_fit_P = temp['event_id_fit_P']
        event_id_fit_S = temp['event_id_fit_S']
        event_id_predict = temp['event_id_predict']

        try:
            final_peak_fit = peak_P_comparison_df[peak_P_comparison_df.event_id.isin(event_id_fit_P)]
            final_peak_predict = peak_P_comparison_df[peak_P_comparison_df.event_id.isin(event_id_predict)]

            gP = plot_prediction_vs_measure_seaborn(final_peak_predict, phase='P', bins=40, vmax=vmax, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gP.ax_joint.loglog(final_peak_fit.peak_P, final_peak_fit.peak_P_predict, 'r.')
            gP.ax_joint.text(2e-3, 50, region_text + f', P wave\n{len(final_peak_fit.dropna())} measurments to fit, \n{len(final_peak_predict.dropna())} measurments to predict', fontsize=15)
            gP.savefig(fig_dir + f'/P_validate_strain_rate_{result_label}_seaborn.png')
        except:
            print('No valid P wave regression results, skip ...')
            pass

        try:
            final_peak_fit = peak_S_comparison_df[peak_S_comparison_df.event_id.isin(event_id_fit_S)]
            final_peak_predict = peak_S_comparison_df[peak_S_comparison_df.event_id.isin(event_id_predict)]
        
            gS = plot_prediction_vs_measure_seaborn(final_peak_predict, phase='S', bins=40, vmax=vmax, xlim=xy_lim, ylim=xy_lim, height=height, space=space)
            gS.ax_joint.loglog(final_peak_fit.peak_S, final_peak_fit.peak_S_predict, 'r.')
            gS.ax_joint.text(2e-3, 50, region_text + f', S wave\n{len(final_peak_fit.dropna())} measurments to fit, \n{len(final_peak_predict.dropna())} measurments to predict', fontsize=15)
            gS.savefig(fig_dir + f'/S_validate_strain_rate_{result_label}_seaborn.png')
        except:
            print('No valid S wave regression results, skip ...')
            pass

    plt.close('all')
    



# %%
