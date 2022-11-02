#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
temp = pd.read_csv('/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/regression_results_smf_weighted_100_channel_at_least/secondary_site_terms_calibration_100chan.csv')
# %%
temp1 = temp.drop_duplicates(subset=['channel_id', 'region'])
# %%
temp_ridgecrest = temp1[temp1.region == 'ridgecrest']
temp_mammothS = temp1[temp1.region == 'mammothS']
temp_mammothN = temp1[temp1.region == 'mammothN']
# %%
fig, ax = plt.subplots(3, 2, sharey=True, figsize=(10, 8))

for i_region, temp_df in enumerate([temp_ridgecrest, temp_mammothN, temp_mammothS]):
    gca = ax[i_region, 0]
    gca.plot(temp_df.channel_id, temp_df.diff_peak_P)
    gca.set_xlabel('channel #')
    gca.set_ylabel('P calibration')
    gca.set_title(temp_df.region.unique())

    gca = ax[i_region, 1]
    gca.plot(temp_df.channel_id, temp_df.diff_peak_S)
    gca.set_xlabel('channel #')
    gca.set_ylabel('S calibration')
    gca.set_title(temp_df.region.unique())

gca.set_ylim(-1e-3, 1e-3)
plt.tight_layout()
# %%
