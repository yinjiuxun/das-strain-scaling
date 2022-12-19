import os
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import urllib3
import requests


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)



# function to get the elevation 
# written by Ettore Biondi
def make_remote_request(url: str, params: dict):
    """
    Makes the remote request
    Continues making attempts until it succeeds
    """

    count = 1
    while True:
        try:
            response = requests.get((url + urllib.parse.urlencode(params)))
        except (OSError, urllib3.exceptions.ProtocolError) as error:
            print('\n')
            print('*' * 20, 'Error Occured', '*' * 20)
            print(f'Number of tries: {count}')
            print(f'URL: {url}')
            print(error)
            print('\n')
            count += 1
            continue
        break

    return response

def elevation_function(lat,lon):
    url = 'https://nationalmap.gov/epqs/pqs.php?'
    params = {'x': lon,
              'y': lat,
              'units': 'Meters',
              'output': 'json'}
    result = make_remote_request(url, params)
    return result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']


event_folder = '/kuafu/EventData/Curie'  #'/kuafu/EventData/AlumRock5.1/MammothNorth'#'/kuafu/EventData/Ridgecrest' 
tt_output_dir = event_folder + '/theoretical_arrival_time0'
