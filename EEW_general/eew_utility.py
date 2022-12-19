import numpy as np
from scipy.interpolate import griddata

def get_DAS_response_time(eq_depth, das_x, das_y, eq_x_grid, eq_y_grid, distance_grid, depth_grid, tavel_time_grid):
    """the ideal response time of DAS system is defined as the earliest S arrival at the array"""
    test_point_x, test_point_y = eq_x_grid.flatten()[np.newaxis,:], eq_y_grid.flatten()[np.newaxis,:]

    # distane from the test point to all the DAS channel (in degree)
    dist_to_das = np.sqrt((das_x[:, np.newaxis] - test_point_x)**2 + (das_y[:, np.newaxis] - test_point_y)**2)/113
    min_distance_to_das = np.min(dist_to_das, axis=0)

    arrival_to_das = griddata(np.array([distance_grid.flatten(), depth_grid.flatten()]).T, tavel_time_grid.flatten(), 
            (min_distance_to_das, np.ones(min_distance_to_das.shape)*eq_depth))

    DAS_response_time = arrival_to_das
    return DAS_response_time


def get_SEIS_response_time(eq_depth, eq_x_grid, eq_y_grid, st_x_grid, st_y_grid, distance_grid, depth_grid, tavel_time_grid):
    """the ideal response time of seismic station is defined at the time when 4 stations detected P wave"""
    test_point_x, test_point_y = eq_x_grid.flatten()[np.newaxis,:], eq_y_grid.flatten()[np.newaxis,:]

    # distane from the test point to all the DAS channel (in degree)
    dist_to_station = np.sqrt((st_x_grid.flatten()[:, np.newaxis] - test_point_x)**2 + (st_y_grid.flatten()[:, np.newaxis] - test_point_y)**2)/113

    n_minimum = 4
    # loop to find the 4th minimum distance 
    dist_to_station_copy = dist_to_station.copy()
    for i_iter in range(n_minimum-1):
        temp_index = np.argmin(dist_to_station_copy, axis=0)
        index_to_discard = tuple(zip(temp_index, range(0, dist_to_station_copy.shape[1])))

        for ii in index_to_discard:
            dist_to_station_copy[ii] = 1e10

    nth_min_dist = np.min(dist_to_station_copy, axis=0)

    arrival_to_sta = griddata(np.array([distance_grid.flatten(), depth_grid.flatten()]).T, tavel_time_grid.flatten(), 
            (nth_min_dist, np.ones(nth_min_dist.shape)*eq_depth))

    STA_response_time = arrival_to_sta
    return STA_response_time