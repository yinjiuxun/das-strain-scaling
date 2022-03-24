import pandas as pd
import numpy as np

def combined_channels(DAS_index, peak_amplitude_df, nearby_channel_number):
    temp1= np.arange(0, DAS_index.max()+1) # original channel number
    temp2 = temp1 // nearby_channel_number # combined channel number
    peak_amplitude_df['combined_channel_id'] = temp2[np.array(peak_amplitude_df.channel_id).astype('int')]
    return peak_amplitude_df


def add_event_label(peak_amplitude_df):
    '''A function to add the event label to the peak ampliutde DataFrame'''
    peak_amplitude_df['event_label'] = 0
    event_id_unique = peak_amplitude_df.event_id.unique()

    for i_event, event_id in enumerate(event_id_unique):
       peak_amplitude_df['event_label'][peak_amplitude_df['event_id'] == event_id] = i_event

    return peak_amplitude_df

#%% Functions used to for the regression
def model_parameters_df(reg, combined_channel_number, digits=3):
    magnitude = round(reg.params[-2],digits)
    distance = round(reg.params[-1],digits)
    magnitude_err = round(np.sqrt(reg.cov_params().iloc[-2][-2]),digits)
    distance_err = round(np.sqrt(reg.cov_params().iloc[-1][-1]),digits)
    parameter_df = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'],
    data = [[combined_channel_number, magnitude, distance, magnitude_err, distance_err]])
    return parameter_df

def fit_regression(combined_channel_number_list, M_threshold, results_output_dir, regression_results_dir, regression_text):
    # DataFrame to store parameters for all models
    P_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err']) 
    S_parameters_comparison = pd.DataFrame(columns=['combined_channels', 'magnitude', 'distance', 'magnitude_err', 'distance_err'])

    for nearby_channel_number in combined_channel_number_list:
        if nearby_channel_number == -1:
            peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_region_site_all.csv')
            peak_amplitude_df_M = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
            regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df_M).fit()
            regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', data=peak_amplitude_df_M).fit()
        else:
            peak_amplitude_df = pd.read_csv(results_output_dir + f'/peak_amplitude_region_site_{nearby_channel_number}.csv')
            peak_amplitude_df_M = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
        # %% Now can fit the data with different regional site terms
            regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_M).fit()
            regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_M).fit()

        regP.save(regression_results_dir + f"/P_regression_region_site_terms_{nearby_channel_number}chan.pickle")
        regS.save(regression_results_dir + f"/S_regression_region_site_terms_{nearby_channel_number}chan.pickle")

    # output to text files
        with open(regression_text + f"/P_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
            text_file.write(regP.summary().as_text())
        with open(regression_text + f"/S_regression_all_events_with_combined_site_terms_{nearby_channel_number}chan.txt", "w") as text_file:
            text_file.write(regS.summary().as_text())

    # Store the parameters 
        P_parameters_comparison = pd.concat([P_parameters_comparison, model_parameters_df(regP, nearby_channel_number)], axis=0)
        S_parameters_comparison = pd.concat([S_parameters_comparison, model_parameters_df(regS, nearby_channel_number)], axis=0)

    P_parameters_comparison.to_csv(regression_text + '/parameter_comparison_P.txt', index=False, sep='\t')
    S_parameters_comparison.to_csv(regression_text + '/parameter_comparison_S.txt', index=False, sep='\t')
