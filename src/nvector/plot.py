"""
Created on 9. des. 2015

@author: pab
"""

import cartopy.feature as cpf
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import numpy as np
from nvector import rad, deg, lat_lon2n_E, unit, n_E2lat_lon


def plot_mean_position():
    """
    Example
    -------
    plot_mean_position()
    Ex7, Average lat=67.2, lon=-6.9
    """
    positions = np.array([(90, 0),
                          (60, 10),
                          (50, -20),
                          ])
    lats, lons = positions.T
    nvecs = lat_lon2n_E(rad(lats), rad(lons))

    # Find the horizontal mean position:
    n_EM_E = unit(np.sum(nvecs, axis=1).reshape((3, 1)))
    lat, lon = n_E2lat_lon(n_EM_E)
    lat, lon = deg(lat), deg(lon)
    print('Ex7, Average lat={0:2.1f}, lon={1:2.1f}'.format(lat[0], lon[0]))

    ax = plt.figure().gca(projection=ccrs.Orthographic(int(lon), int(lat)))
    ax.add_feature(cpf.OCEAN, zorder=0)
    ax.add_feature(cpf.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cpf.COASTLINE)
    ax.add_feature(cpf.BORDERS, linestyle=':')
    ax.add_feature(cpf.LAKES, alpha=0.5)
    ax.add_feature(cpf.RIVERS)

    ax.set_global()
    ax.gridlines()

    # vector_crs = ccrs.Geodetic()
    vector_crs = ccrs.PlateCarree()

    ax.scatter(lon, lat, linewidth=5, marker='o', color='r',
               transform=vector_crs)
    ax.scatter(lons, lats, linewidth=5, marker='o', color='k',
               transform=vector_crs)

    plt.title('Figure of mean position (red dot) compared to \npositions '
              'A, B, and C (black dots).')


if __name__ == '__main__':
    from nvector._common import test_docstrings
    test_docstrings(__file__)
    plot_mean_position()
    plt.show('hold')
