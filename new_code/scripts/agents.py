"""
File containing functions for agents/particles.
"""
import numpy as np

from forces import allignment_force, error_force, contact_force, part_repulsive_force
from environment import periodic_boundaries
from utils import rescale
from constants import alpha, beta, gamma, N, delta_t, mass_par, dimensions, v_mag, bound_cond

# # force parrameters
# alpha = 0 # stregnth of repulsive force between to the particles
# beta = 0 # stregnth of the force due to the objects on the particles
# gamma = 1 # stregnth of allignment
#
# N = 2  # number of particles
# delta_t = 1     # time increment
# mass_par = 1 # masss of the particles
# dimensions = 2
# v_mag = 0.05      # total magnitude of each particle velocity
#
#
# bound_cond = True

def update_system(positions, velocities, positions_obj, polygons):
    """
    Updates the positons and velocities of ALL the particles in a system.
    """
    # lists which will contain the updated values
    new_positions = []
    new_vels = []

    # loop through each index in the positions, vel, acc
    for i in range(N):
        # get the acceleration based on the positions of the particles
        acceleration = update_acceleration(positions[i], velocities[i], positions, velocities, positions_obj, polygons)
        # call update to get the new value
        new_vel = update_velocity(velocities[i], acceleration)
        new_pos = update_position(positions[i], new_vel, polygons)

        # print("particles: {}".format(i))
        # print("position: {}".format(new_pos))
        # print("velocity: {}".format(new_vel))
        # print("\n")

        # append it to the new values
        new_positions.append(new_pos)
        new_vels.append(new_vel)

    return new_positions, new_vels

def update_position(position, velocity, polygons):
    """
    Update the location of a particle and returns the new location.
    """
    # create a new lsit which will contain the new position
    new_pos = []

    # loop through the dimensions in position
    for i in range(dimensions):

        # add the velocity in that dimension to the position (times delta_t)
        pos_i = position[i] + velocity[i] * delta_t

        # chek for boundry conditions
        if bound_cond == True:
            pos_i = periodic_boundaries(pos_i)

        # append to the new_position and velocity list this position/velocity
        new_pos.append(pos_i)


    return new_pos

def update_velocity(velocity, acceleration):
    """
    Update the velocity of a particle and returns the new velocity.
    """
    # create a new lsit which will contain the new position
    new_vel = []

    # loop through the dimensions in position
    for i in range(dimensions):
        # update the velocity first
        v_i = velocity[i] + acceleration[i] * delta_t

        # append to the new_position and velocity list this position/velocity
        new_vel.append(v_i)

    # rescale the magnitude of the speed
    new_vel = rescale(v_mag, new_vel)

    # add the noise
    # new_vel = error_force(new_vel)

    return new_vel

def update_acceleration(position_particle, velocity_particle, position_particles, velocity_particles, positions_obj, polygons):
    """
    Algorithm which updates the algorithm
    """
    # define two inital forces dependent on the particles and on hte object
    force_object = np.array([0., 0.])
    force_particles = np.array([0., 0.])

    # loop through each particle and calculate the repulsive force from the particle
    for particle in position_particles:
        if particle == position_particle:
            continue
        force_particles += part_repulsive_force(position_particle, particle)

    # calcualte force due to the objects
    for object in range(len(positions_obj)):
        force_object -= contact_force(polygons[object], positions_obj[object], position_particle, velocity_particle)


    new_acceleration = (alpha * force_particles + gamma * allignment_force(position_particle, velocity_particle, position_particles, velocity_particles) ) / mass_par
    # new_acceleration += error_force(velocity_particle + new_acceleration*delta_t) / mass_par
    new_acceleration += (beta * force_object) / mass_par

    # print(new_acceleration)
    return new_acceleration
