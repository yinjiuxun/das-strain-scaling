# %%
import os
import numpy as np
import datetime
import sys
from matplotlib import pyplot as plt

# import the Obspy modules that we will use in this exercise
import obspy
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import pandas as pd

sys.path.append('../')
from utility.general import mkdir
from utility.loading import save_rawevent_h5, load_event_data
# %%
def download_station(network, station, channel, location, path):
    fname = path + "/" + network + "." + station + "." + location + "." + channel
    inventory = client.get_stations(network=network, station=station, channel=channel,
                                    location=location, level="response")
    inventory.write(fname + ".xml", format="STATIONXML")
    return inventory


# download waveform given network
def download_waveforms(network, station, channel, location, starttime, endtime, path, fname):
    tr = client.get_waveforms(network=network, station=station, channel=channel,
                              location=location, starttime=starttime - 1800, endtime=endtime + 1800,
                              attach_response=True)
    # tr.detrend("spline", order=3, dspline=1000)
    # tr.remove_response(output="VEL")

    # here to deal with the taperring at both end, only keep the central 1-hour long data
    newtr = tr.slice(starttime, endtime)  # probably fix later

    newtr.write(path + "/" + fname + ".mseed", format='MSEED')
    return newtr


# %%
client = Client('IRIS', debug=True)

#%% Specify the network and stations we want.
networks = np.array(['C1'])  # A specific seismic network to which the stations belong
stations = np.array(['VA01'])  # Names of the stations
channels = np.array(['BHE'])  # Channels
location = '*' #'10'

results_output_dir = '/kuafu/yinjx/Curie/seimsic_data'
mkdir(results_output_dir)

waveform_dir = results_output_dir + '/continuous_waveforms'
mkdir(waveform_dir)

station_dir = waveform_dir + '/stations'
mkdir(station_dir)

for network in networks:
    for station in stations:
        for channel in channels:
            inventory = download_station(network, station, channel, location, station_dir)

        # %%
sta_lat = inventory[0][0].latitude
sta_lon = inventory[0][0].longitude

# %% Search for the events within a given range
t1 = obspy.UTCDateTime("2022-06-10")
t2 = obspy.UTCDateTime("2022-06-15")

# Download data for individual stations
network = inventory.networks[0]
stations = network.stations
for station in stations:
    network_code = network.code
    station_code = station.code
    print('=' * 12 + network_code + '.' + station_code + '=' * 12)

    fname = network_code + "." + station_code + "." + location + "." + t1.strftime("%Y%m%d") + "-" + t2.strftime("%Y%m%d")
    try:
        tr = download_waveforms(network_code, station_code, 'BHE', location, t1, t2, waveform_dir, fname)
    except:
        print("Issue downloading data from " + network_code + '.' + station_code)
# %%

tr = obspy.read(f'{waveform_dir}/{fname}.mseed')

t0 = obspy.UTCDateTime("2022-06-14T3:32:52")

tr_work = tr.trim(starttime=t0, endtime=t0+60)
#%%
tr_work.decimate(int(tr_work[0].stats.sampling_rate/20)) # donwsample to 10 Hz
tr_work.filter('highpass', freq=0.5) # this filtering is actually optional, please feel free to try

# tr_work.plot(type='dayplot', interval=60*12, size=(800, 1200))


tr_work.plot()

# %%
# compare the seismic data with the DAS data
event_folder = '/kuafu/EventData/Curie'
catalog = pd.read_csv(event_folder + '/catalog.csv')
time_drift = {'9000':12.5, '9001':12.5, '9002':0, '9003':0, '9004':9, '9005':8.5, '9006':9, '9007':9}
ymin_list = [0, 0, 0]
for ii in [1]:
    event_now = catalog.iloc[ii, :]
    event_id = event_now.event_id
    event_name = event_now.place
    magnitude = event_now.magnitude

    data, info = load_event_data(event_folder + '/data_raw', event_id)
    DAS_channel_num = data.shape[1]
    dt = info['dt_s']
    das_time = np.arange(data.shape[0])*dt-30

    t0=obspy.UTCDateTime(info['event_time']) + 4*3600


# %%
t0=obspy.UTCDateTime(info['event_time']) + 4*3600
tr_work = tr.trim(starttime=t0, endtime=t0+60)

dt = 1/tr_work[0].stats.sampling_rate
tr_time = np.arange(0, tr_work[0].stats.npts)*dt

plt.plot(tr_time, tr_work[0].data)
plt.plot(tP, 0, 'rx')
plt.plot(tS, 0, 'rx')
plt.xlim(0, 30)

# %%
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

model = TauPyModel(model='iasp91')
distance_to_source = locations2degrees(sta_lat, sta_lon, 
            catalog.iloc[ii,:].latitude, catalog.iloc[ii,:].longitude)

arrivals = model.get_ray_paths(catalog.iloc[ii,:].depth_km, distance_to_source, phase_list=['p', 's'])
tP = arrivals[0].time
tS = arrivals[1].time 

plt.plot(das_time[::10], tr_work[0].data[:-1])