#%% import modules
import pandas as pd
#from sep_util import read_file
import numpy as np
import pygmt

import xarray
#%%
# =========================  Plot both arrays in Chile with PyGMT ==============================
gmt_region = [-72.5, -71, -33.2, -31.8]
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=gmt_region)
# %%
bathymetry = grid.values
lon = grid.coords['lon']
lat = grid.coords['lat']
# %%
