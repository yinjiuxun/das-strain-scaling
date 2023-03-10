# das_strain_scaling
Code to reproduce results of manuscript "Earthquake magnitude with DAS: a transferable data-based scaling relation"

Data can be downloaded from Caltech DATA: https://doi.org/10.22002/rxtp0-38405

The scripts for regression are in the folder ./regression
Scripts for results validation and visualization are in the folder ./validation_prediction

## 1. Download the data from Caltech DATA: https://doi.org/10.22002/rxtp0-38405, upzip it and rename to data_files in the current directory

## 2. Run the Python script in regression to get the results
- ### iter_regression.py gives the regression results from the given data sets: (1) combined dataset of 3 California arrays; (2) individual array of Ridgecrest, Long Valley North and Long Valley South, Sanriku

- ### The results will be put to the new directories named: iter_results, iter_results_Ridgecrest, iter_results_LongValley_N, iter_results_LongValley_S and iter_results_Sanriku

- ### transfer_regression.py gives the transfered regression results: using the coefficients from the combined dataset, and the measurements from 5 randomly chosen events from Sanriku dataset to calibrate the site terms.

## 3. The scripts and notebooks in the directory validation_prediction can reproduce the figures of the paper

- ### check_peak_amplitude_info.ipynb: notebook to reproduce Figure 1 and 2
- ### magnitude_estimation.py: script to reproduce Figure 3
- ### real_time_estimation.py: script to reproduce Figure 4
- ### Others scripts can reproduce Figures in the Supporting Information  