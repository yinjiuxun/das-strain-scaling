#%% import modules
import pandas as pd
import numpy as np
from obspy.geodetics import locations2degrees

# %%
def filter_events_with_scaling(catalog, das_info, region=None,
    M_coef=(0.437, 0.69), D_coef=(-1.2693, -1.5875), detectable_amplitude=None, mean_site_term=None):
    """ 
    catalog = filter_events_with_scaling(catalog, das_info, region=None,
    M_coef=(0.437, 0.69), D_coef=(-1.2693, -1.5875), detectable_amplitude=None, mean_site_term=None)
    
    region can be one from ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']

    if region is not specified, detectable_amplitude (minimum detectable amplitude at the instrument) 
    and mean_site_term (averaged site term of an array) have to be given
    
    """
    
    detectable_amplitude_known = {'ridgecrest': -0.875, 'mammothN': -1.964, 'mammothS': -1.796, 'sanriku': -2.304}
    mean_S_site_term_known = {'ridgecrest': 0.299, 'mammothN': 0.449, 'mammothS': 0.459, 'sanriku': -0.374}
    mean_P_site_term_known = {'ridgecrest': 0.401, 'mammothN': 0.574, 'mammothS': 0.250, 'sanriku': np.nan}

    if region in ['ridgecrest', 'mammothN', 'mammothS', 'sanriku']:
        detectable_amplitude = 10**detectable_amplitude_known[region]
        mean_site_term = 10**mean_S_site_term_known[region]

    # distance from events to das array
    das_center_lon = das_info.longitude.mean()
    das_center_lat = das_info.latitude.mean()

    distance_list = locations2degrees(das_center_lat, das_center_lon, catalog.latitude, catalog.longitude) * 113


    # get the scaling relation
    D_sense_P = 10**((-(catalog.magnitude)*M_coef[0] + (np.log10(detectable_amplitude)-np.log10(mean_site_term)))/D_coef[0])
    D_sense_S = 10**((-(catalog.magnitude)*M_coef[1] + (np.log10(detectable_amplitude)-np.log10(mean_site_term)))/D_coef[1])

    return catalog[distance_list<D_sense_S]



#%%
catalog = pd.read_csv('/kuafu/EventData/Mammoth_south/catalog.csv')
das_info = pd.read_csv('/kuafu/EventData/Mammoth_south/das_info.csv')
catalog_filter = filter_events_with_scaling(catalog, das_info, region='mammothS')

# %%
