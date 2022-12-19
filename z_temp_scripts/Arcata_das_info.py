#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#%%
# DAS location "/kuafu/DASdata/DASinfo/DAS_ChannelLocation/DAS-LAX_coor_interp.csv"
# Write DAS info
DAS_info = pd.read_csv('/kuafu/EventData/Arcata_Spring2022/Lat-Lon-Channels_V2_2022_08_21.csv')

# 4780 channels in the data file, but 4790 in the DAS info...Calibrate that...
DAS_channel_num = int(DAS_info['Channel# at 2m'].max() - DAS_info['Channel# at 2m'].min() + 1)

# The incorrect lat lon files...
points = np.array([DAS_info['RTK GPS Location (Long) (50cm-accuracy)'],
                   DAS_info['RTK GPS Location (Lat) (50cm-accuracy)']]).T  # a (nbre_points x nbre_dim) array

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

# Interpolation with slinear:
interpolations_method = 'slinear'
alpha = np.linspace(DAS_channel_num, 0, DAS_channel_num)/DAS_channel_num

interpolator =  interp1d(distance, points, kind=interpolations_method, axis=0)
interpolated_points = interpolator(alpha)

DAS_index = np.arange(int(DAS_info['Channel# at 2m'].min()), int(DAS_info['Channel# at 2m'].max())+1)
DAS_lon = interpolated_points[:, 0]
DAS_lat = interpolated_points[:, 1]

# Write to a csv in the EventData folder
DAS_info_df = pd.DataFrame({'index': DAS_index, 'latitude': DAS_lat, 'longitude': DAS_lon, 'elevation_m': 0.0})
DAS_info_df.to_csv('/kuafu/EventData/Arcata_Spring2022/das_info.csv', index=False)


# %%
