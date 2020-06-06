"""
Rotor code that contain the objects/rotors/polygons for the simualtion.
"""

from utils import centroid
from forces import torque_force
from constants import L, b, M, delta_t, mass_object, mom_inertia, dimensions, beta

import numpy as np
import math

# # constants
# L = 5 # size of the box
# b = L * 0.2
# M = 1   # number of objects
# delta_t = 1     # time increment
# beta = 0  # strenght of the force due to the objects on the particles
# mass_object = 100 # masss of the object
# mom_inertia = (1/3) * mass_object
# dimensions = 2


def polygon(origin, a, angle_diff, spikes):
    """
    Define the polygon from the points on the verticies.
    """
    refvec = (0, 1)

    def clockwiseangle_and_distance(point):
        """
        Finds angle between point and ref vector ffor sortinf points in rotor
        in order of angles
        """


        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]

        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])

        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0

        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2

        angle = math.atan2(diffprod, dotprod)

        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle
        # I return first the angle because that's the primary sorting criterium
        return angle


    pts_in = []
    pts_out = []
    pts_tot = []

    x_0, y_0 = origin  # centre of circle


    for i in range(spikes):
        value_out = [x_0 + b * np.cos(2 * np.pi * i / spikes), y_0 + b * np.sin(2 * np.pi * i / spikes)]
        value_in = [x_0 + a * np.cos(angle_diff + 2 * np.pi * i / spikes), y_0 + a * np.sin(angle_diff + 2 * np.pi * i / spikes)]
        pts_out.append(value_out)
        pts_in.append(value_in)

    sorted_out = sorted(pts_out, key=clockwiseangle_and_distance)
    sorted_in = sorted(pts_in, key=clockwiseangle_and_distance)

    for i in range(len(sorted_in)):
        pts_tot.append(sorted_out[i])
        pts_tot.append(sorted_in[i])

    return pts_tot


def objects(polygons):
    """
    Create a set of M objects, defining them just by there centre of mass.
    As of now they are basically particles of different species.
    """

    # will hold each posi/vel/acc for each particle in the system
    positions = []
    ang_velocities = []
    accelerations = []

    for i in range(M):
        # lsit containing positions and velocities at random
        init_position = centroid(polygons[i])
        init_velocity = 0
        init_acceleration = [0 for i in range(dimensions)]

        # append the positions to the bigger lists
        positions.append(init_position)
        ang_velocities.append(init_velocity)
        accelerations.append(init_acceleration)

    return positions, ang_velocities, accelerations

# ----------------------- Update Functions for objects ------------------------------

def update_system_object(polygons, positions_obj, ang_velocities_obj, position_particles, velocity_particles):
    """
    Updates the positons and velocities of ALL the objects in a system.
    """
    # lists which will contain the updated values
    new_ang_vels = []
    new_polygons = []

    # loop through each index in the positions, vel, acc
    for i in range(len(positions_obj)):

        # # get the new torque on the force
        # torque = update_torque_object(polygons[i], positions_obj[i], velocity_particles, position_particles)
        # new_pol = update_velocity_object()
        # update the anngular acceleration on the object
        ang_acceleration = update_ang_acceleration_object(polygons[i], positions_obj[i], ang_velocities_obj[i], positions_obj, position_particles, velocity_particles)
        # update the angular velocity of the object
        new_ang_vel = update_ang_velocity_object(ang_velocities_obj[i], ang_acceleration)
        # update the position of the vertex
        new_vers = update_position_object_vertex(polygons[i], positions_obj[i], new_ang_vel)

        # append them to the list of new position
        new_ang_vels.append(new_ang_vel)
        new_polygons.append(new_vers)

    return new_polygons, new_ang_vels

def update_ang_velocity_object(velocities_obj, accelerations_obj):
    """
    Update the velocity of a particle and returns the new velocity.
    """

    # create a new lsit which will contain the new position
    new_ang_vel = velocities_obj + accelerations_obj * delta_t

    return new_ang_vel

def update_ang_acceleration_object(polygon, position_obj, ang_vel_object, positions_obj, position_particles, velocity_particles):
    """
    Algorithm which updates the acceleration of the com of the object
    """
    # define two inital forces dependent on the particles and on hte object
    torque_particles = 0

    # loop through each particle and calculate the repulsive force from the particle
    for particle in range(len(position_particles)):
        torque_particles += torque_force(polygon, position_obj,ang_vel_object, velocity_particles[particle],position_particles[particle])

    new_acceleration = beta * torque_particles / mom_inertia

    return new_acceleration

def update_position_object_vertex(polygon, position_obj, ang_vel_object):
    """
    Update the location of a particle and returns the new location.
    """
    polygon = np.array(polygon)

    new_pos = []
    # get the change in angle from the angular velocuty
    angle = ang_vel_object * delta_t

    # get the relative positions of the polygon, i.e with respective to the centre of the polygon
    polygon_respective = [(polygon[i] - position_obj).tolist() for i in range(polygon.shape[0])]
    # print(polygon_respective)

    # build the rotation matrix
    rot_mat = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    # print(rot_mat)
    # print(polygon_respective[0])

    # multiply by old points in polygon
    for i in range(polygon.shape[0]):
        new_pos_i = np.dot(rot_mat, polygon_respective[i]) + position_obj
        new_pos.append(new_pos_i.tolist())

    return new_pos
