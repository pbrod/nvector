'''
Created on 9. des. 2015

@author: pab
'''
from mpl_toolkits.basemap import Basemap  # @UnresolvedImport
import matplotlib.pyplot as plt
import numpy as np
from nvector import (set_north_pole_axis_for_E_frame, rad, deg, lat_lon2n_E,
                     unit, n_E2lat_lon)


R_Ee = set_north_pole_axis_for_E_frame(axis='z')


def plot_mean_position():
    positions = np.array([(90, 0),
                          (60, 10),
                          (50, -20),
                          (70,-70)])
    lats, lons = positions.T
    nvecs = lat_lon2n_E(rad(lats), rad(lons))

    # Find the horizontal mean position:
    n_EM_E = unit(np.sum(nvecs, axis=1).reshape((3, 1)))
    lat, lon = n_E2lat_lon(n_EM_E)
    lat, lon = deg(lat), deg(lon)
    print('Ex7, Average lat={}, lon={}'.format(lat, lon))

    map1 = Basemap(projection='ortho', lat_0=int(lat), lon_0=int(lon),
                   resolution='l')
    plot_world(map1)
    x, y = map1(lon, lat)
    map1.scatter(x, y, linewidth=5, marker='o', color='r')

    x1, y1 = map1(lons, lats)
    print(len(lons), x1, y1)
    map1.scatter(x1, y1, linewidth=5, marker='o', color='k')

    plt.title('Figure of mean position (red dot) compared to positions '
              'A, B, and C (black dots).')
    plt.show('hold')


def plot_world(map1):
    """
    Parameters
    ----------
    map1: Basemap object
        map1 to plot.
    """
    map1.drawcoastlines(linewidth=0.25)
    map1.drawcountries(linewidth=0.25)
    map1.fillcontinents(color='coral', lake_color='aqua', alpha=0.25)
    map1.drawmapboundary(fill_color='aqua')
    map1.drawmeridians(np.arange(0, 360, 30))
    map1.drawparallels(np.arange(-90, 90, 30))


if __name__ == '__main__':
    plot_mean_position()
