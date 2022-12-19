import pandas as pd
import numpy as np
import dateutil
import h5py

# Functions to save event h5 files
def save_rawevent_h5(fn, data, info):
   """
   """
   info_copy = info.copy()
   with h5py.File(fn, 'w') as fid:
       fid.create_dataset('data', data=data)
       for key in info.keys():
           if isinstance(info[key], str):
               #fid['data'].attrs.modify(key, np.string_(info_copy[key]))
               fid['data'].attrs.modify(key, info_copy[key])
           else:
               fid['data'].attrs.modify(key, info_copy[key])

def load_event_data(data_path, eq_id):
    with h5py.File(data_path + '/' + str(eq_id) + '.h5', 'r') as fid:
        data = fid['data'][:]
        info = {}
        for key in fid['data'].attrs.keys():
            info[key] = fid['data'].attrs[key]
        if 'begin_time' in info.keys():
            info['begin_time'] = dateutil.parser.parse(info['begin_time'])
        if 'end_time' in info.keys():
            info['end_time'] = dateutil.parser.parse(info['end_time'])
        if 'event_time' in info.keys():
            info['event_time'] = dateutil.parser.parse(info['event_time'])
    return data, info


def load_phasenet_pick(pick_path, eq_id, das_time, channel, time_range=None, include_nan=False):
    picks = pd.read_csv(pick_path + f'/{eq_id}.csv')

    picks_P = picks[picks.phase_type == 'P'].drop_duplicates(subset=['channel_index'], keep='first')
    picks_S = picks[picks.phase_type == 'S'].drop_duplicates(subset=['channel_index'], keep='first')


    # Adding restriction on the time
    if time_range is not None:
        dt = das_time[1] - das_time[0]
        picks_P = picks_P[(picks_P.phase_type == 'P') & 
                            (picks_P.phase_index <= time_range[1]/dt) & 
                            (picks_P.phase_index >= time_range[0]/dt)]

        picks_S = picks_S[(picks_S.phase_type == 'S') & 
                            (picks_S.phase_index <= time_range[3]/dt) & 
                            (picks_S.phase_index >= time_range[2]/dt)]

    if include_nan:
        picks_P_time = np.ones(channel.shape) * np.nan
        picks_S_time = np.ones(channel.shape) * np.nan
        ii_p = channel.isin(picks_P.channel_index.unique())#picks_P.channel_index.isin(channel)
        ii_s = channel.isin(picks_S.channel_index.unique())#picks_S.channel_index.isin(channel)

        ii_p_picks = picks_P.channel_index.isin(channel)
        ii_s_picks = picks_S.channel_index.isin(channel)

        picks_P_time[ii_p] = das_time[picks_P.phase_index[ii_p_picks]]
        picks_S_time[ii_s] = das_time[picks_S.phase_index[ii_s_picks]]
        channel_P, channel_S = channel, channel

    else:
        picks_P_time = das_time[picks_P.phase_index]
        channel_P = channel[picks_P.channel_index]

        picks_S_time = das_time[picks_S.phase_index]
        channel_S = channel[picks_S.channel_index]

    return picks_P_time, channel_P, picks_S_time, channel_S



import scipy.io as spio
def read_mat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _tolist(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)