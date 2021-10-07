# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - PRESENT  Zhengyu Peng
# E-mail: zpeng.me@gmail.com
# Website: https://zpeng.me

# `                      `
# -:.                  -#:
# -//:.              -###:
# -////:.          -#####:
# -/:.://:.      -###++##:
# ..   `://:-  -###+. :##:
#        `:/+####+.   :##:
# .::::::::/+###.     :##:
# .////-----+##:    `:###:
#  `-//:.   :##:  `:###/.
#    `-//:. :##:`:###/.
#      `-//:+######/.
#        `-/+####/.
#          `+##+.
#           :##:
#           :##:
#           :##:
#           :##:
#           :##:
#            .+:


cimport cython

from libc.math cimport sin, cos, sqrt, atan, atan2, acos, pow, fmax, M_PI
from libcpp cimport bool

from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t, vector
from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.radarsimc cimport Target, PointCloud

import numpy as np
cimport numpy as np
# from stl import mesh
import meshio

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lidar_scene(lidar, targets, t=0):
    """
    lidar_scene(lidar, targets, t=0)

    Lidar scene simulator

    :param dict lidar:
        Lidar configuration

        {

        - **position** (*numpy.1darray*) --
            Lidar's position (m). [x, y, z]
        - **phi** (*numpy.1darray*) --
            Array of phi scanning angles (deg) 
        - **theta** (*numpy.1darray*) --
            Array of theta scanning angles (deg)

        }

    :param list[dict] targets:
        Target list

        [{

        - **model** (*str*) --
            Path to the target model
        - **origin** (*numpy.1darray*) --
            Origin position of the target model (m), [x, y, z].
            ``default [0, 0, 0]``
        - **location** (*numpy.1darray*) --
            Location of the target (m), [x, y, z].
            ``default [0, 0, 0]``
        - **speed** (*numpy.1darray*) --
            Speed of the target (m/s), [vx, vy, vz].
            ``default [0, 0, 0]``
        - **rotation** (*numpy.1darray*) --
            Target's angle (deg), [yaw, pitch, roll].
            ``default [0, 0, 0]``
        - **rotation_rate** (*numpy.1darray*) --
            Target's rotation rate (deg/s),
            [yaw rate, pitch rate, roll rate]
            ``default [0, 0, 0]``

        }]

    :param time t:
        Simulation time stampes. ``default 0``

    :return: rays
    :rtype: numpy.array
    """
    cdef PointCloud[float_t] pointcloud
    
    cdef float_t[:,:] points_memview
    cdef uint64_t[:,:] cells_memview
    cdef float_t[:] origin
    cdef float_t[:] speed
    cdef float_t[:] location
    cdef float_t[:] rotation
    cdef float_t[:] rotation_rate
    
    cdef int_t target_count = len(targets)
    cdef int_t idx
    
    for idx in range(0, target_count):
        t_mesh = meshio.read(targets[idx]['model'])

        points_memview = t_mesh.points.astype(np.float64)
        cells_memview = t_mesh.cells[0].data.astype(np.uint64)

        origin = np.array(targets[idx].get('origin', (0,0,0)), dtype=np.float64)

        location = np.array(targets[idx].get('location', (0,0,0)), dtype=np.float64)+t*np.array(targets[idx].get('speed', (0,0,0)), dtype=np.float64)
        speed = np.array(targets[idx].get('speed', (0,0,0)), dtype=np.float64)

        rotation = np.array(targets[idx].get('rotation', (0,0,0)), dtype=np.float64)/180*np.pi+t*np.array(targets[idx].get('rotation_rate', (0,0,0)), dtype=np.float64)/180*np.pi
        rotation_rate = np.array(targets[idx].get('rotation_rate', (0,0,0)), dtype=np.float64)/180*np.pi

        pointcloud.AddTarget(Target[float_t](&points_memview[0,0],
            &cells_memview[0,0],
            <int_t> cells_memview.shape[0],
            Vec3[float_t](&origin[0]),
            Vec3[float_t](&location[0]),
            Vec3[float_t](&speed[0]),
            Vec3[float_t](&rotation[0]),
            Vec3[float_t](&rotation_rate[0]),
            <bool> targets[idx].get('is_ground', False)))

    cdef float_t[:] phi = np.array(lidar['phi'], dtype=np.float64)/180*np.pi
    cdef float_t[:] theta = np.array(lidar['theta'], dtype=np.float64)/180*np.pi

    cdef vector[float_t] phi_vector
    phi_vector.reserve(phi.shape[0])
    cdef vector[float_t] theta_vector
    theta_vector.reserve(theta.shape[0])

    for idx in range(0, phi.shape[0]):
        phi_vector.push_back(phi[idx])

    for idx in range(0, theta.shape[0]):
        theta_vector.push_back(theta[idx])
    
    pointcloud.Sbr(
        phi_vector,
        theta_vector,
        Vec3[float_t](<float_t> lidar['position'][0], <float_t> lidar['position'][1], <float_t> lidar['position'][2])
    )

    ray_type = np.dtype([('positions', np.float64, (3,)), ('directions', np.float64, (3,))])

    rays = np.zeros(pointcloud.cloud_.size(), dtype=ray_type)

    for idx in range(0, pointcloud.cloud_.size()):
        rays[idx]['positions'][0] = pointcloud.cloud_[idx].loc_[1][0]
        rays[idx]['positions'][1] = pointcloud.cloud_[idx].loc_[1][1]
        rays[idx]['positions'][2] = pointcloud.cloud_[idx].loc_[1][2]
        rays[idx]['directions'][0] = pointcloud.cloud_[idx].dir_[1][0]
        rays[idx]['directions'][1] = pointcloud.cloud_[idx].dir_[1][1]
        rays[idx]['directions'][2] = pointcloud.cloud_[idx].dir_[1][2]
    
    return rays
