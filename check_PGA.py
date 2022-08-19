#%% import modules
import os
import glob
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# import the plotting functions
from plotting_functions import *
# import the utility functions
from utility_functions import *

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# %%
# load the PGA data from NGA
pga_dir = '/kuafu/yinjx/PGA_scaling/Updated_NGA_West2_flatfiles_part1'
# files = glob.glob(pga_dir + '/*.xlsx')
files = glob.glob(pga_dir + '/Updated_NGA_West2_Flatfile_RotD50_d050_public_version.xlsx')
# %%
PGA_df = pd.DataFrame(columns=['EQID','Station Sequence Number', 'PGA (g)', 'T0.100S', 'T0.200S', 'T1.000S', 'Magnitude Type', 'Earthquake Magnitude', 'HypD (km)'])
# for file in files:
file = files[0]
temp_df = pd.read_excel(file)
PGA_df = pd.concat([PGA_df, temp_df[['EQID','Station Sequence Number', 'PGA (g)', 'T0.100S', 'T0.200S', 'T1.000S', 'Magnitude Type', 'Earthquake Magnitude', 'HypD (km)']]])
# %%
PGA_df.to_csv(pga_dir + '/PGA_for_regression.csv', index=False)
# %%
# load the dataframe
PGA_df = pd.read_csv(pga_dir + '/PGA_for_regression.csv')
PGA_df = PGA_df[PGA_df['Magnitude Type'] == 'Mw']

PGA_df = PGA_df.rename(columns={"Station Sequence Number": "station_id",
                       "PGA (g)": "PGA_g",
                       "T0.100S": "PSA_10Hz",
                       "T0.200S": "PSA_5Hz",
                       "T1.000S": "PSA_1Hz",
                       "Earthquake Magnitude": "magnitude",
                       "HypD (km)": "distance"})

# visualize the data
PGA_df['log10(distance)'] = np.log10(PGA_df['distance'].astype('float'))
PGA_df['log10(PGA)'] = np.log10(PGA_df['PGA_g'].astype('float'))
PGA_df['log10(PSA_10Hz)'] = np.log10(PGA_df['PSA_10Hz'].astype('float'))
PGA_df['log10(PSA_5Hz)'] = np.log10(PGA_df['PSA_5Hz'].astype('float'))
PGA_df['log10(PSA_1Hz)'] = np.log10(PGA_df['PSA_1Hz'].astype('float'))

cmap = plt.cm.get_cmap('Spectral_r', 5)
fig, ax = plt.subplots(2,2, figsize=(10, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax = ax.flatten()
gca=ax[0]
clb = gca.scatter(PGA_df['log10(distance)'], PGA_df['log10(PGA)'], c=PGA_df['magnitude'], s=1, alpha=1,vmax=8, vmin=3, cmap=cmap)
gca.set_xlabel('log10(hypo-distance)')
gca.set_ylabel('log10(PGA)')


gca=ax[1]
clb = gca.scatter(PGA_df['log10(distance)'], PGA_df['log10(PSA_10Hz)'], c=PGA_df['magnitude'], s=1, alpha=1,vmax=8, vmin=3, cmap=cmap)
gca.set_xlabel('log10(hypo-distance)')
gca.set_ylabel('log10(PSA_10Hz)')

gca=ax[2]
clb = gca.scatter(PGA_df['log10(distance)'], PGA_df['log10(PSA_5Hz)'], c=PGA_df['magnitude'], s=1, alpha=1,vmax=8, vmin=3, cmap=cmap)
gca.set_xlabel('log10(hypo-distance)')
gca.set_ylabel('log10(PSA_5Hz)')

gca=ax[3]
clb = gca.scatter(PGA_df['log10(distance)'], PGA_df['log10(PSA_1Hz)'], c=PGA_df['magnitude'], s=1, alpha=1,vmax=8, vmin=3, cmap=cmap)
gca.set_xlabel('log10(hypo-distance)')
gca.set_ylabel('log10(PSA_1Hz)')

plt.savefig(pga_dir + '/NGA_PGA_PGS_figure.png')
#fig.colorbar(clb, orientation="vertical", label='Magnitude')
# %%
# apply the same regression to the PGA without site
reg_site = smf.ols(formula='np.log10(PGA_g) ~ magnitude + np.log10(distance) + C(station_id) - 1', data=PGA_df).fit()
PSA_10Hz_reg_site = smf.ols(formula='np.log10(PSA_10Hz) ~ magnitude + np.log10(distance) + C(station_id) - 1', data=PGA_df).fit()
PSA_5Hz_reg_site = smf.ols(formula='np.log10(PSA_5Hz) ~ magnitude + np.log10(distance) + C(station_id) - 1', data=PGA_df).fit()
PSA_1Hz_reg_site = smf.ols(formula='np.log10(PSA_1Hz) ~ magnitude + np.log10(distance) + C(station_id) - 1', data=PGA_df).fit()
# regP = smf.wls(formula='np.log10(PGA_g) ~ magnitude + np.log10(distance)-1', data=PGA_df, weights=(10**(PGA_df.magnitude*0.6))).fit()

# print the scaling parameters
print(reg_site.params[-2:])
print(PSA_10Hz_reg_site.params[-2:])
print(PSA_5Hz_reg_site.params[-2:])
print(PSA_1Hz_reg_site.params[-2:])

reg_site.save(pga_dir + '/PGA_OLS.pickle', remove_data=True)
PSA_10Hz_reg_site.save(pga_dir + '/PSA_10Hz_OLS.pickle', remove_data=True)
PSA_5Hz_reg_site.save(pga_dir + '/PSA_5Hz_OLS.pickle', remove_data=True)
PSA_1Hz_reg_site.save(pga_dir + '/PSA_1Hz_OLS.pickle', remove_data=True)
#%%
# apply the same regression to the PGA with site
reg_site_weight = smf.wls(formula='np.log10(PGA_g) ~ magnitude + np.log10(distance) + C(station_id) - 1', data=PGA_df, weights=10**(PGA_df.magnitude)).fit()
PSA_10Hz_reg_site_weight = smf.wls(formula='np.log10(PSA_10Hz) ~ magnitude + np.log10(distance) + C(station_id) - 1', data=PGA_df, weights=10**(PGA_df.magnitude)).fit()
PSA_5Hz_reg_site_weight = smf.wls(formula='np.log10(PSA_5Hz) ~ magnitude + np.log10(distance) + C(station_id) - 1', data=PGA_df, weights=10**(PGA_df.magnitude)).fit()
PSA_1Hz_reg_site_weight = smf.wls(formula='np.log10(PSA_1Hz) ~ magnitude + np.log10(distance) + C(station_id) - 1', data=PGA_df, weights=10**(PGA_df.magnitude)).fit()
# regP = smf.wls(formula='np.log10(PGA_g) ~ magnitude + np.log10(distance)-1', data=PGA_df, weights=(10**(PGA_df.magnitude*0.6))).fit()

# print the scaling parameters
print(reg_site_weight.params[-2:])
print(PSA_10Hz_reg_site_weight.params[-2:])
print(PSA_5Hz_reg_site_weight.params[-2:])
print(PSA_1Hz_reg_site_weight.params[-2:])

reg_site_weight.save(pga_dir + '/PGA_WLS.pickle', remove_data=True)
PSA_10Hz_reg_site_weight.save(pga_dir + '/PSA_10Hz_WLS.pickle', remove_data=True)
PSA_5Hz_reg_site_weight.save(pga_dir + '/PSA_5Hz_WLS.pickle', remove_data=True)
PSA_1Hz_reg_site_weight.save(pga_dir + '/PSA_1Hz_WLS.pickle', remove_data=True)
# %%
