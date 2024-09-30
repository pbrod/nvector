"""
Created on 9. des. 2015

@author: pab
"""
from functools import partial
try:
    import matplotlib.pyplot as plt
except (ImportError, OSError, ModuleNotFoundError):
    plt = None
try:
    import cartopy.feature as cpf
    import cartopy.crs as ccrs
except (ImportError, OSError, ModuleNotFoundError):
    cpf = ccrs = None

import numpy as np
from nvector import rad, deg, lat_lon2n_E, unit, n_E2lat_lon


def _init_earth_plotter(lat, lon):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(int(lon), int(lat)))
    ax.add_feature(cpf.OCEAN, zorder=0)
    ax.add_feature(cpf.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cpf.COASTLINE)
    ax.add_feature(cpf.BORDERS, linestyle=':')
    ax.add_feature(cpf.LAKES, alpha=0.5)
    ax.add_feature(cpf.RIVERS)
    ax.set_global()
    ax.gridlines()
    # Alternatively: ccrs.Geodetic()
    vector_crs = ccrs.PlateCarree()
    return partial(ax.scatter, transform=vector_crs)


def _init_plotter(lat, lon):
    if ccrs:  # Cartopy did load
        return _init_earth_plotter(lat, lon)
    ax = plt.figure().gca()
    return ax.scatter


def plot_mean_position():
    """
    Example
    -------
    >>> plot_mean_position()
    Ex7, Average lat=67.2, lon=-6.9
    >>> plt.show()  # doctest: +SKIP
    >>> plt.close()
    """
    positions = np.array([(90, 0),
                          (60, 10),
                          (50, -20),
                          ])
    lats, lons = np.transpose(positions)
    nvecs = lat_lon2n_E(rad(lats), rad(lons))

    # Find the horizontal mean position:
    n_EM_E = unit(np.sum(nvecs, axis=1).reshape((3, 1)))
    lat, lon = n_E2lat_lon(n_EM_E)
    lat, lon = deg(lat), deg(lon)
    print('Ex7, Average lat={0:2.1f}, lon={1:2.1f}'.format(lat[0], lon[0]))

    plotter = _init_plotter(lat, lon)

    plotter(lon, lat, linewidth=5, marker='o', color='r')
    plotter(lons, lats, linewidth=5, marker='o', color='k')

    plt.title('Figure of mean position (red dot) compared to \npositions '
              'A, B, and C (black dots).')


def main():
    rotated_crs = ccrs.RotatedPole(pole_longitude=180.0, pole_latitude=90.0)

    ax = plt.axes(projection=rotated_crs)
    # ax.set_extent([-6, 3, 48, 58], crs=ccrs.PlateCarree())
    ax.set_extent([9, 12, 58, 60], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.show()


if __name__ == '__main__':

    from nvector._common import test_docstrings
    test_docstrings(__file__)
