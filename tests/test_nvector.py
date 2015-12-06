'''
This file contains solutions to the examples given at
www.navlab.net/nvector

The content of this file is based on the following publication:

Gade, K. (2010). A Nonsingular Horizontal Position Representation, The Journal
of Navigation, Volume 63, Issue 03, pp 395-417, July 2010.
(www.navlab.net/Publications/A_Nonsingular_Horizontal_Position_Representation.pdf)

Copyright (c) 2015, Norwegian Defence Research Establishment (FFI)
All rights reserved.

Originated: 2015.03.26 Kenneth Gade, FFI

NOTES:
- All angles are by default assumed to be in radians, if an angle is
in degrees, the variable name has the following ending: _deg

- The dot product (inner product) of vectors x and y is written dot(x,y)
here to make the code more readable for those unfamiliar with
Matlab. In Matlab one would normally write x'*y (i.e. x transposed
times y)
'''
import numpy as np
import unittest
from nvector import (deg, rad,
                     lat_long2n_E,
                     n_E2lat_long,
                     n_E2R_EN,
                     zyx2R, unit,
                     n_EA_E_and_n_EB_E2p_AB_E,
                     n_EA_E_and_p_AB_E2n_EB_E)
from nvector._core import p_EB_E2n_EB_E, n_EB_E2p_EB_E, R_Ee
from numpy.testing import assert_array_almost_equal


class TestNvector(unittest.TestCase):

    def test_A_and_B_to_delta(self):

        # Positions A and B are given in (decimal) degrees and depths:

        # Position A:
        lat_EA_deg = 1
        long_EA_deg = 2
        z_EA = 3

        # Position B:
        lat_EB_deg = 4
        long_EB_deg = 5
        z_EB = 6

        # Find the exact vector between the two positions, given in meters
        # north, east, and down, i.e. find p_AB_N.

        # SOLUTION:
        # Step1: Convert to n-vectors (rad() converts to radians):
        n_EA_E = lat_long2n_E(rad(lat_EA_deg), rad(long_EA_deg))
        n_EB_E = lat_long2n_E(rad(lat_EB_deg), rad(long_EB_deg))

        # Step2: Find p_AB_E (delta decomposed in E).
        # WGS-84 ellipsoid is default:
        p_AB_E = n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA, z_EB)

        # Step3: Find R_EN for position A:
        R_EN = n_E2R_EN(n_EA_E)

        # Step4: Find p_AB_N
        p_AB_N = np.dot(R_EN.T, p_AB_E)
        # (Note the transpose of R_EN: The "closest-rule" says that when
        # decomposing, the frame in the subscript of the rotation matrix that
        # is closest to the vector, should equal the frame where the vector is
        # decomposed. Thus the calculation R_NE*p_AB_E is correct, since the
        # vector is decomposed in E, and E is closest to the vector. In the
        # example we only had R_EN, and thus we must transpose it: R_EN'=R_NE)

        # Step5: Also find the direction (azimuth) to B, relative to north:
        azimuth = np.arctan2(p_AB_N[1], p_AB_N[0])
        # positive angle about down-axis

        print('Ex1, delta north, east, down = {}, {}, {}'.format(p_AB_N[0],
                                                                 p_AB_N[1],
                                                                 p_AB_N[2]))
        print('Ex1, azimuth = {} deg'.format(deg(azimuth)))

        self.assertAlmostEqual(p_AB_N[0], 331730.23478089)
        self.assertAlmostEqual(p_AB_N[1], 332997.87498927)
        self.assertAlmostEqual(p_AB_N[2], 17404.27136194)
        self.assertAlmostEqual(deg(azimuth), 45.10926324)

    def test_B_and_delta_to_C(self):
        # delta vector from B to C, decomposed in B is given:
        p_BC_B = np.r_[3000, 2000, 100].reshape((-1, 1))

        # Position and orientation of B is given:
        n_EB_E = unit([[1], [2], [3]])  # unit to get unit length of vector
        z_EB = -400
        R_NB = zyx2R(rad(10), rad(20), rad(30))
        # the three angles are yaw, pitch, and roll

        # A custom reference ellipsoid is given (replacing WGS-84):
        a = 6378135
        f = 1 / 298.26  # (WGS-72)

        # Find the position of C.
        # SOLUTION:
        # Step1: Find R_EN:
        R_EN = n_E2R_EN(n_EB_E)

        # Step2: Find R_EB, from R_EN and R_NB:
        R_EB = np.dot(R_EN, R_NB)  # Note: closest frames cancel

        # Step3: Decompose the delta vector in E:
        p_BC_E = np.dot(R_EB, p_BC_B)
        # no transpose of R_EB, since the vector is in B

        # Step4: Find the position of C, using the functions that goes from one
        # position and a delta, to a new position:
        n_EC_E, z_EC = n_EA_E_and_p_AB_E2n_EB_E(n_EB_E, p_BC_E, z_EB, a, f)

        # When displaying the resulting position for humans, it is more
        # convenient to see lat, long:
        lat_EC, long_EC = n_E2lat_long(n_EC_E)
        # Here we also assume that the user wants output height (= - depth):
        print('Ex2, Pos C: lat, long = {},{} deg,  height = {} m'.format(deg(lat_EC),
                                                                         deg(long_EC),
                                                                         -z_EC))

        self.assertAlmostEqual(deg(lat_EC), 53.32637826)
        self.assertAlmostEqual(deg(long_EC), 63.46812344)
        self.assertAlmostEqual(z_EC, -406.00719607)

    def test_ECEF_vector_to_geodetic_latitude(self):

        # Position B is given as p_EB_E ("ECEF-vector")

        p_EB_E = 6371e3 * np.vstack((0.9, -1, 1.1))  # m

        # Find position B as geodetic latitude, longitude and height

        # SOLUTION:

        # Find n-vector from the p-vector:
        n_EB_E, z_EB = p_EB_E2n_EB_E(p_EB_E)

        # Convert to lat, long and height:

        lat_EB, long_EB = n_E2lat_long(n_EB_E)
        h_EB = -z_EB

        print('Ex3, Pos B: lat, long = {} {} deg, height = {} m'.format(deg(lat_EB),
                                                                        deg(long_EB),
                                                                        h_EB))

        self.assertAlmostEqual(deg(lat_EB), 39.37874867)
        self.assertAlmostEqual(deg(long_EB), -48.0127875)
        self.assertAlmostEqual(h_EB, 4702059.83429485)

    def test_geodetic_latitude_to_ECEF_vector(self):

        # Position B is given with lat, long and height:
        lat_EB_deg = 1
        long_EB_deg = 2
        h_EB = 3

        # Find the vector p_EB_E ("ECEF-vector")

        # SOLUTION:
        # Step1: Convert to n-vector:
        n_EB_E = lat_long2n_E(rad(lat_EB_deg), rad(long_EB_deg))

        # Step2: Find the ECEF-vector p_EB_E:
        p_EB_E = n_EB_E2p_EB_E(n_EB_E, -h_EB)

        print('Ex4: p_EB_E = {} m'.format(p_EB_E.ravel()))

        assert_array_almost_equal(p_EB_E.ravel(),
                                  [6373290.27721828, 222560.20067474,
                                   110568.82718179])

    def test_surface_distance(self):

        # Position A and B are given as n_EA_E and n_EB_E:
        # Enter elements directly:
        # n_EA_E=unit([1 0 -2]')
        # n_EB_E=unit([-1 -2 0]')

        # or input as lat/long in deg:
        n_EA_E = lat_long2n_E(rad(88), rad(0))
        n_EB_E = lat_long2n_E(rad(89), rad(-170))

        r_Earth = 6371e3  # m, mean Earth radius

        # SOLUTION:

        # The great circle distance is given by equation (16) in Gade (2010):
        # Well conditioned for all angles:
        s_AB = np.arctan2(np.linalg.norm(np.cross(n_EA_E, n_EB_E, axis=0), axis=0),
                          np.dot(n_EA_E.T, n_EB_E)) * r_Earth

        # # ill conditioned for small angels:
        # s_AB_version1=acos(dot(n_EA_E,n_EB_E))*r_Earth
        #
        # # ill-conditioned for angles near pi/2 (and not valid above pi/2)
        # s_AB_version2=asin(norm(cross(n_EA_E,n_EB_E)))*r_Earth

        # The Euclidean distance is given by:
        d_AB = np.linalg.norm(n_EB_E - n_EA_E, axis=0) * r_Earth

        print(
            'Ex5, Great circle distance = {} km, Euclidean distance = {} km'.format(
                s_AB /
                1000,
                d_AB /
                1000))

        self.assertAlmostEqual(s_AB / 1000, 332.45644411)
        self.assertAlmostEqual(d_AB / 1000, 332.41872486)

    def test_interpolated_position(self):

        # Position B at time t0 and t2 is given as n_EB_E_t0 and n_EB_E_t1:
        # Enter elements directly:
        # n_EB_E_t0=unit([1 0 -2]')
        # n_EB_E_t1=unit([-1 -2 0]')

        # or input as lat/long in deg:
        n_EB_E_t0 = lat_long2n_E(rad(89), rad(0))
        n_EB_E_t1 = lat_long2n_E(rad(89), rad(180))

        # The times are given as:
        t0 = 10
        t1 = 20
        ti = 16  # time of interpolation

        # Find the interpolated position at time ti, n_EB_E_ti

        # SOLUTION:

        # Using standard interpolation:
        n_EB_E_ti = unit(
            n_EB_E_t0 + (ti - t0) * (n_EB_E_t1 - n_EB_E_t0) / (t1 - t0))

        # When displaying the resulting position for humans, it is more
        # convenient to see lat, long:
        lat_EB_ti, long_EB_ti = n_E2lat_long(n_EB_E_ti)
        print('Ex6, Interpolated position: lat, long = {} {} deg'.format(deg(lat_EB_ti),
                                                                         deg(long_EB_ti)))

        self.assertAlmostEqual(deg(lat_EB_ti), 89.7999805)
        self.assertAlmostEqual(deg(long_EB_ti), 180.)

    def test_mean_position(self):

        # Three positions A, B and C are given:
        # Enter elements directly:
        # n_EA_E=unit([1 0 -2]')
        # n_EB_E=unit([-1 -2 0]')
        # n_EC_E=unit([0 -2 3]')

        # or input as lat/long in deg:
        n_EA_E = lat_long2n_E(rad(90), rad(0))
        n_EB_E = lat_long2n_E(rad(60), rad(10))
        n_EC_E = lat_long2n_E(rad(50), rad(-20))

        # Find the horizontal mean position:
        n_EM_E = unit(n_EA_E + n_EB_E + n_EC_E)

        # The result is best viewed with a figure that shows the n-vectors
        # relative to an Earth-model:
        print('Ex7, See figure')
        # plot_Earth_figure(n_EA_E,n_EB_E,n_EC_E,n_EM_E)
        assert_array_almost_equal(n_EM_E.ravel(),
                                  [0.384117, -0.046602, 0.922107])

    def test_position_A_and_azimuth_and_distance_to_B(self):

        # Position A is given as n_EA_E:
        # Enter elements directly:
        # n_EA_E=unit([1 0 -2]')

        # or input as lat/long in deg:
        n_EA_E = lat_long2n_E(rad(80), rad(-90))

        # The initial azimuth and great circle distance (s_AB), and Earth
        # radius (r_Earth) are also given:
        azimuth = rad(200)
        s_AB = 1000  # m
        r_Earth = 6371e3  # m, mean Earth radius

        # Find the destination point B, as n_EB_E ("The direct/first geodetic
        # problem" for a sphere)

        # SOLUTION:

        # Step1: Find unit vectors for north and east:
        k_east_E = unit(np.cross(np.dot(R_Ee().T, [[1], [0], [0]]),
                                 n_EA_E, axis=0))
        k_north_E = np.cross(n_EA_E, k_east_E, axis=0)

        # Step2: Find the initial direction vector d_E:
        d_E = k_north_E * np.cos(azimuth) + k_east_E * np.sin(azimuth)

        # Step3: Find n_EB_E:
        n_EB_E = n_EA_E * np.cos(s_AB / r_Earth) + d_E * np.sin(s_AB / r_Earth)

        # When displaying the resulting position for humans, it is more
        # convenient to see lat, long:
        lat_EB, long_EB = n_E2lat_long(n_EB_E)
        print('Ex8, Destination: lat, long = {} {} deg'.format(deg(lat_EB),
                                                               deg(long_EB)))
        self.assertAlmostEqual(deg(lat_EB), 79.99154867)
        self.assertAlmostEqual(deg(long_EB), -90.01769837)

    def test_intersection(self):

        # Two paths A and B are given by two pairs of positions:
        # Enter elements directly:
        # n_EA1_E=unit([1 0 -2]')
        # n_EA2_E=unit([-1 -2 0]')
        # n_EB1_E=unit([0 -2 3]')
        # n_EB2_E=unit([0 -2 3]')

        # or input as lat/long in deg:
        n_EA1_E = lat_long2n_E(rad(10), rad(20))
        n_EA2_E = lat_long2n_E(rad(30), rad(40))
        n_EB1_E = lat_long2n_E(rad(50), rad(60))
        n_EB2_E = lat_long2n_E(rad(70), rad(80))

        # Find the intersection between the two paths, n_EC_E:
        n_EC_E_tmp = unit(np.cross(np.cross(n_EA1_E, n_EA2_E, axis=0),
                                   np.cross(n_EB1_E, n_EB2_E, axis=0), axis=0))

        # n_EC_E_tmp is one of two solutions, the other is -n_EC_E_tmp. Select
        # the one that is closet to n_EA1_E, by selecting sign from the dot
        # product between n_EC_E_tmp and n_EA1_E:
        n_EC_E = np.sign(np.dot(n_EC_E_tmp.T, n_EA1_E)) * n_EC_E_tmp

        # When dipslaying the resulting position for humans, it is more
        # convenient to see lat, long:
        lat_EC, long_EC = n_E2lat_long(n_EC_E)
        print(
            'Ex9, Intersection: lat, long = {} {} deg'.format(
                deg(lat_EC),
                deg(long_EC)))
        self.assertAlmostEqual(deg(lat_EC), 40.31864307)
        self.assertAlmostEqual(deg(long_EC), 55.90186788)

    def test_cross_track_distance(self):

        # Position A1 and A2 and B are given as n_EA1_E, n_EA2_E, and n_EB_E:
        # Enter elements directly:
        # n_EA1_E=unit([1 0 -2]')
        # n_EA2_E=unit([-1 -2 0]')
        # n_EB_E=unit([0 -2 3]')

        # or input as lat/long in deg:
        n_EA1_E = lat_long2n_E(rad(0), rad(0))
        n_EA2_E = lat_long2n_E(rad(10), rad(0))
        n_EB_E = lat_long2n_E(rad(1), rad(0.1))

        r_Earth = 6371e3  # m, mean Earth radius

        # Find the cross track distance from path A to position B.

        # SOLUTION:

        # Find the unit normal to the great circle:
        c_E = unit(np.cross(n_EA1_E, n_EA2_E, axis=0)).T

        # Find the great circle cross track distance:
        s_xt = (np.arccos(np.dot(c_E, n_EB_E)) - np.pi / 2) * r_Earth

        # Find the Euclidean cross track distance:
        d_xt = -np.dot(c_E, n_EB_E) * r_Earth

        print(
            'Ex10, Cross track distance = {} m, Euclidean = {} m'.format(
                s_xt,
                d_xt))

        self.assertAlmostEqual(s_xt, 11117.79911015)
        self.assertAlmostEqual(d_xt, 11117.79346741)


#
# function plot_Earth_figure(n_EA_E,n_EB_E,n_EC_E,n_EM_E)
# # Plotting the Earth figure for Example 7.
#
# Earth_radius_for_plotting =0.7 # plot an Earth sphere with radius smaller
# # than 1 (n-vector) to make the tip of the n-vectors visible.
#
# # R_Ee selects correct E-axes, see R_Ee.m for details:
# n_EA_E=R_Ee*n_EA_E n_EB_E=R_Ee*n_EB_E n_EC_E=R_Ee*n_EC_E n_EM_E=R_Ee*n_EM_E
#
# figure(1)
# clf
# hold on
#
# # To plot 3D arrows the function arrow3d.m, available from Matlab file
# # exchange, written by Moshe Lindner is used (a copy of the function is at
# # the end of this file):
# arrow3d([0 -n_EA_E(3)],[0 n_EA_E(2)],[0 n_EA_E(1)],0.9,0.02,0.05,'r') # A, n_EA_E
# arrow3d([0 -n_EB_E(3)],[0 n_EB_E(2)],[0 n_EB_E(1)],0.9,0.02,0.05,'r') # B, n_EB_E
# arrow3d([0 -n_EC_E(3)],[0 n_EC_E(2)],[0 n_EC_E(1)],0.9,0.02,0.05,'r') # C, n_EC_E
# arrow3d([0 -n_EM_E(3)],[0 n_EM_E(2)],[0 n_EM_E(1)],0.9,0.02,0.05,'g') # M, n_EM_E
#
#
# ###################### Plotting a spherical Earth surface:
# try
#     # Loads a simple topographic model of Earth (available as part of default Matlab)
#     load topo
#
#     # Remove height info, only storing info about water or land:
#     Earth_topo_binary=zeros(size(topo))
#     Earth_topo_binary(topo>0)=1 # Set all positions above water to 1
#     Earth_topo_binary(topo<=0)=-1 # Set all positions below or equal to zero to -1
#     Earth_topo_binary=[Earth_topo_binary(:,181:360) Earth_topo_binary(:,1:180)] # Switch the halves to get correct mapping for our plot
#     clear topo
# catch
#     Earth_topo_binary=zeros(2)-1
# end
#
# # Number of elements in the sphere (in each direction)
# n_of_Earth_plot_elements=90
#
# # Data for a 3D Earth sphere
# [Earth_plot_X,Earth_plot_Y,Earth_plot_Z]=sphere(n_of_Earth_plot_elements)
# Earth_plot_X=Earth_plot_X*Earth_radius_for_plotting
# Earth_plot_Y=Earth_plot_Y*Earth_radius_for_plotting
# Earth_plot_Z=Earth_plot_Z*Earth_radius_for_plotting
#
# Earth_surface_properties.Cdata = Earth_topo_binary # Color data
# Earth_surface_properties.FaceLighting = 'gouraud' # Smooth out light reflex
# Earth_surface_properties.FaceColor= 'texture' # Note that the 'facecolor' property needs to be set to
# # 'texturemap' if the size of the z-data is different from the size of the data in the colormap (topo) that is loaded.
# Earth_surface_properties.EdgeColor = 'none' # Remove mesh
# Earth_surface_properties.AmbientStrength=0.1 # Ambient light strength
# surface(Earth_plot_X,Earth_plot_Y,Earth_plot_Z,Earth_surface_properties)
# material dull
#
# colmap_Earth_binary=[0.6 0.6 1
#     0.6 1 0.6] # blue and green, coloring sea and land
# colormap(colmap_Earth_binary)
#
# # Add two light sources:
# light('Position',[0 -1 0.5],'Style','infinite')
# light('Position',[0 1 -0.5],'Style','infinite')
#
# hold off
# grid on
# axis equal
# title('Example 7')
# view(60,30)
# rotate3d on
#
# end
#
#
# # The following code, written by Moshe Lindner, is from Matlab file exchange, to plot 3D arrows:
#
# function [h]=arrow3d(x,y,z,head_frac,radii,radii2,colr)
# #
# # The function plotting 3-dimensional arrow
# #
# # h=arrow3d(x,y,z,head_frac,radii,radii2,colr)
# #
# # The inputs are:
# #       x,y,z =  vectors of the starting point and the ending point of the
# #           arrow, e.g.:  x=[x_start, x_end] y=[y_start, y_end]z=[z_start,z_end]
# #       head_frac = fraction of the arrow length where the head should  start
# #       radii = radius of the arrow
# #       radii2 = radius of the arrow head (defult = radii*2)
# #       colr =   color of the arrow, can be string of the color name, or RGB vector  (default='blue')
# #
# # The output is the handle of the surfaceplot graphics object.
# # The settings of the plot can changed using: set(h, 'PropertyName', PropertyValue)
# #
# # example #1:
# #        arrow3d([0 0],[0 0],[0 6],.5,3,4,[1 0 .5])
# # example #2:
# #        arrow3d([2 0],[5 0],[0 -6],.2,3,5,'r')
# # example #3:
# #        h = arrow3d([1 0],[0 1],[-2 3],.8,3)
# #        set(h,'facecolor',[1 0 0])
# #
# # Written by Moshe Lindner , Bar-Ilan University, Israel.
# # July 2010 (C)
#
# if nargin==5
#     radii2=radii*2
#     colr='blue'
# elseif nargin==6
#     colr='blue'
# end
# if size(x,1)==2
#     x=x'
#     y=y'
#     z=z'
# end
#
# x(3)=x(2)
# x(2)=x(1)+head_frac*(x(3)-x(1))
# y(3)=y(2)
# y(2)=y(1)+head_frac*(y(3)-y(1))
# z(3)=z(2)
# z(2)=z(1)+head_frac*(z(3)-z(1))
# r=[x(1:2)',y(1:2)',z(1:2)']
#
# N=50
# dr=diff(r)
# dr(end+1,:)=dr(end,:)
# origin_shift=(ones(size(r))*(1+max(abs(r(:))))+[dr(:,1) 2*dr(:,2) -dr(:,3)])
# r=r+origin_shift
#
# normdr=(sqrt((dr(:,1).^2)+(dr(:,2).^2)+(dr(:,3).^2)))
# normdr=[normdr,normdr,normdr]
# dr=dr./normdr
# Pc=r
# n1=cross(dr,Pc)
# normn1=(sqrt((n1(:,1).^2)+(n1(:,2).^2)+(n1(:,3).^2)))
# normn1=[normn1,normn1,normn1]
# n1=n1./normn1
# P1=n1+Pc
#
# X1=[]Y1=[]Z1=[]
# j=1
# for theta=([0:N])*2*pi./(N)
#     R1=Pc+radii*cos(theta).*(P1-Pc) + radii*sin(theta).*cross(dr,(P1-Pc)) -origin_shift
#     X1(2:3,j)=R1(:,1)
#     Y1(2:3,j)=R1(:,2)
#     Z1(2:3,j)=R1(:,3)
#     j=j+1
# end
#
# r=[x(2:3)',y(2:3)',z(2:3)']
#
# dr=diff(r)
# dr(end+1,:)=dr(end,:)
# origin_shift=(ones(size(r))*(1+max(abs(r(:))))+[dr(:,1) 2*dr(:,2) -dr(:,3)])
# r=r+origin_shift
#
# normdr=(sqrt((dr(:,1).^2)+(dr(:,2).^2)+(dr(:,3).^2)))
# normdr=[normdr,normdr,normdr]
# dr=dr./normdr
# Pc=r
# n1=cross(dr,Pc)
# normn1=(sqrt((n1(:,1).^2)+(n1(:,2).^2)+(n1(:,3).^2)))
# normn1=[normn1,normn1,normn1]
# n1=n1./normn1
# P1=n1+Pc
#
# j=1
# for theta=([0:N])*2*pi./(N)
#     R1=Pc+radii2*cos(theta).*(P1-Pc) + radii2*sin(theta).*cross(dr,(P1-Pc)) -origin_shift
#     X1(4:5,j)=R1(:,1)
#     Y1(4:5,j)=R1(:,2)
#     Z1(4:5,j)=R1(:,3)
#     j=j+1
# end
#
# X1(1,:)=X1(1,:)*0 + x(1)
# Y1(1,:)=Y1(1,:)*0 + y(1)
# Z1(1,:)=Z1(1,:)*0 + z(1)
# X1(5,:)=X1(5,:)*0 + x(3)
# Y1(5,:)=Y1(5,:)*0 + y(3)
# Z1(5,:)=Z1(5,:)*0 + z(3)
#
# h=surf(X1,Y1,Z1,'facecolor',colr,'edgecolor','none')
# #camlight
# lighting phong
#
# end
#


if __name__ == "__main__":
    # import syssys.argv = ['', 'Test.testName']
    unittest.main()
