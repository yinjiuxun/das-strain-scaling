# %%
# import modules
import pandas as pd
import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import remove_outliers
from utility.regression import store_regression_results, fit_regression_iteration

def main(data_file, results_output_dir, weighted='wls', wavetypes=['P', 'S'], 
         M_threshold=[2, 10], snr_threshold=10, min_channel=100, n_iter=20, rms_epsilon=0.1):
    
    # Set parameters
    outlier_value = 1e4

    # Create output directory
    mkdir(results_output_dir)

    # Load data
    peak_amplitude_dataframe = load_peak_amplitude_data(data_file, outlier_value)

    ##########################
    if peak_amplitude_dataframe['region'].unique()[0] == 'sanriku': # some special processing for Sanriku data
        peak_amplitude_dataframe = peak_amplitude_dataframe.drop(index=peak_amplitude_dataframe[peak_amplitude_dataframe.event_id == 4130].index)
    ##########################

    # Fit regressions
    regression_results = {}
    site_terms_dataframe = pd.DataFrame(columns=['region', 'channel_id', 'site_term_P', 'site_term_S'])
    for wavetype in wavetypes:
        try:
            regression, site_terms = fit_regression_iteration(peak_amplitude_dataframe, wavetype=wavetype, weighted=weighted,
                                                               M_threshold=M_threshold, snr_threshold=snr_threshold,
                                                               min_channel=min_channel, n_iter=n_iter, rms_epsilon=rms_epsilon)
            regression_results[wavetype] = regression
            # Store results
            store_regression_results(regression_results[wavetype], results_output_dir, results_filename=f"/{wavetype}_regression_combined_site_terms_iter")
            
            site_terms['wavetype'] = wavetype
            # site_terms_dataframe = site_terms_dataframe.append(site_terms, ignore_index=True)
            site_terms_dataframe = pd.concat((site_terms_dataframe, site_terms), ignore_index=True)

        except Exception as e:
            print(f"{wavetype} regression failed with error: {e}")

    # Store site terms
    site_terms_dataframe.to_csv(f"{results_output_dir}/site_terms_iter.csv", index=False)


def load_peak_amplitude_data(data_file, outlier_value):
    with open(f"{data_file}", 'r') as f:
        peak_amplitude_dataframe = pd.read_csv(f)

    # use hypocentral distance instead of epicentral distance
    peak_amplitude_dataframe['distance_in_km'] = peak_amplitude_dataframe['calibrated_distance_in_km']

    # remove columns not needed
    peak_amplitude_dataframe = peak_amplitude_dataframe[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km',
                                                         'snrP', 'snrS', 'peak_P', 'peak_S', 'region']]
    
    # remove some outliers
    peak_amplitude_dataframe = remove_outliers(peak_amplitude_dataframe, outlier_value=outlier_value)
    peak_amplitude_dataframe = peak_amplitude_dataframe.drop(index=peak_amplitude_dataframe[peak_amplitude_dataframe.peak_P <= 0].index)
    peak_amplitude_dataframe = peak_amplitude_dataframe.drop(index=peak_amplitude_dataframe[peak_amplitude_dataframe.peak_S <= 0].index)

    # A special case when there is other quality control QA
    if 'QA' in peak_amplitude_dataframe.columns:
        peak_amplitude_dataframe = peak_amplitude_dataframe[peak_amplitude_dataframe.QA == 'Yes']

    return peak_amplitude_dataframe

if __name__ == '__main__':   
    # First apply to the combined dataset of the three CA array
    # setup the directory of data and results
    data_file = '../data_files/peak_amplitude/peak_amplitude_multiple_arrays.csv'
    results_output_dir = '../iter_results'
    mkdir(results_output_dir)
    # run the iterative regression
    main(data_file, results_output_dir)

    # Next apply to the individual array
    data_file_list = ['../data_files/peak_amplitude/peak_amplitude_Ridgecrest.csv',
                      '../data_files/peak_amplitude/peak_amplitude_LongValley_N.csv',
                      '../data_files/peak_amplitude/peak_amplitude_LongValley_S.csv',
                      '../data_files/peak_amplitude/peak_amplitude_Sanriku.csv']  

    results_output_dir_list = ['../iter_results_Ridgecrest',
                               '../iter_results_LongValley_N',
                               '../iter_results_LongValley_S',
                               '../iter_results_Sanriku']  
    
    # setup the directory of data and results
    for i_region in range(len(data_file_list)):
        data_file = data_file_list[i_region]
        results_output_dir = results_output_dir_list[i_region]
        mkdir(results_output_dir)

        if 'Sanriku' in data_file:
            snr_threshold = 5 # for Sanriku, we loose the SNR requirement, or there will be too few events to get the scaling relation
        else:
            snr_threshold = 10

        # run the iterative regression
        print("Working on " + data_file)
        main(data_file, results_output_dir, snr_threshold=snr_threshold, min_channel=100)
