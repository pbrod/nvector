'''
Unittests for the rotation module
'''
from functools import partial
import numpy as np
from nvector.util import rad
from nvector.rotation import (xyz2R,
                              R2xyz,
                              zyx2R,
                              R2zyx,
                              n_E_and_wa2R_EL,
                              n_E2R_EN,
                              R_EN2n_E,
                              R_EL2n_E,
                              )


from numpy.testing import assert_allclose as _assert_allclose  # @UnresolvedImport

assert_allclose = partial(_assert_allclose, atol=1e-15)


def test_R2xyz_with_vectors():
    x, y, z = rad(((10, 10), (20, 20), (30, 30)))
    R_AB1 = xyz2R(x, y, z)
    R_AB = np.array([[0.81379768, -0.46984631, 0.34202014],
                     [0.54383814, 0.82317294, -0.16317591],
                     [-0.20487413, 0.31879578, 0.92541658]])[:, :, None]
    R_AB = np.concatenate((R_AB, R_AB), axis=2)
    assert_allclose(R_AB, R_AB1)
    x1, y1, z1 = R2xyz(R_AB1)
    assert_allclose((x, y, z), (x1, y1, z1))


def test_R2xyz():
    x, y, z = rad((10, 20, 30))
    R_AB1 = xyz2R(x, y, z)
    R_AB = [[0.81379768, -0.46984631, 0.34202014],
            [0.54383814, 0.82317294, -0.16317591],
            [-0.20487413, 0.31879578, 0.92541658]]
    assert_allclose(R_AB, R_AB1)
    x1, y1, z1 = R2xyz(R_AB1)
    assert_allclose((x, y, z), (x1, y1, z1))


def test_R2zxy_0():
    x, y, z = rad((0, 0, 0))
    R_AB1 = zyx2R(z, y, x)
    # print(R_AB1.tolist())
    R_AB = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]

    assert_allclose(R_AB, R_AB1)
    z1, y1, x1 = R2zyx(R_AB1)
    assert_allclose((x, y, z), (x1, y1, z1))


def test_R2zxy_z90():
    x, y, z = rad((0, 0, 90))
    R_AB1 = zyx2R(z, y, x)
    # print(R_AB1.tolist())
    R_AB = [[0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]]

    assert_allclose(R_AB, R_AB1)
    z1, y1, x1 = R2zyx(R_AB1)
    assert_allclose((x, y, z), (x1, y1, z1))


def test_R2zxy_y90():
    x, y, z = rad((0, 90, 0))
    R_AB1 = zyx2R(z, y, x)
    # print(R_AB1.tolist())
    R_AB = [[0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]]

    assert_allclose(R_AB, R_AB1)
    z1, y1, x1 = R2zyx(R_AB1)
    assert_allclose((x, y, z), (x1, y1, z1))


def test_R2zxy_x90():
    x, y, z = rad((90, 0, 0))
    R_AB1 = zyx2R(z, y, x)
    # print(R_AB1.tolist())
    R_AB = [[1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]]

    assert_allclose(R_AB, R_AB1)
    z1, y1, x1 = R2zyx(R_AB1)
    assert_allclose((x, y, z), (x1, y1, z1))


def test_R2zxy():
    x, y, z = rad((10, 20, 30))
    R_AB1 = zyx2R(z, y, x)
    # print(R_AB1.tolist())
    R_AB = [[0.8137976813493738, -0.44096961052988237, 0.37852230636979245],
            [0.46984631039295416, 0.8825641192593856, 0.01802831123629725],
            [-0.3420201433256687, 0.16317591116653482, 0.9254165783983234]]

    assert_allclose(R_AB, R_AB1)
    z1, y1, x1 = R2zyx(R_AB1)
    assert_allclose((x, y, z), (x1, y1, z1))


def test_n_E_and_wa2R_EL():
    n_E = np.array([[0], [0], [1]])
    R_EL = n_E_and_wa2R_EL(n_E, wander_azimuth=np.pi / 2)
    R_EL1 = [[0, -1.0, 0],
             [-1.0, 0, 0],
             [0, 0, -1.0]]
    assert_allclose(R_EL, R_EL1)

    R_EN = n_E2R_EN(n_E)
    assert_allclose(R_EN, np.diag([-1, 1, -1]))

    n_E1 = R_EL2n_E(R_EN)
    n_E2 = R_EN2n_E(R_EN)
    assert_allclose(n_E, n_E1)
    assert_allclose(n_E, n_E2)
