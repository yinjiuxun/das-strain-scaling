#%%
import glob
import pandas as pd
import numpy as np


# %%
file_dir = '/kuafu/EventData/Ridgecrest/theoretical_arrival_time'
files = glob.glob(file_dir + '/1D_tt_*.csv')
# %%
for tt_file in files:
    eq_id = tt_file[-12:-4]
    temp_pd = pd.read_csv(tt_file)
    temp_pd = temp_pd[['index','P_arrival','S_arrival']]
    temp_pd = temp_pd.rename(columns={'index':'ichan','P_arrival':'tp','S_arrival':'ts'})
    temp_pd['tp'] = temp_pd['tp'] - 30
    temp_pd['ts'] = temp_pd['ts'] - 30
    temp_pd.to_csv(file_dir + '/' + eq_id + '.table')
# %%
