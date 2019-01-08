"""
Program that inserts lots of particles in box and makes them allign
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# global variables used in the program
L = 15     # size of the box
delta_t = 1     # time increment
dimensions = 2   # dimensions
N = 5    # number of particles

def main():
    """
    Execution of main program.
    """
    global L, delta_t, dimensions

    # positions and velcities over time
    pos_over_t = []
    vel_over_t = []

    # populate the box and append them to pos_over_t and vel_over_t
    locations, velocities = pop_box()
    pos_over_t.append(locations)
    vel_over_t.append(velocities)

    # update position of each particle in the box
    for i in range(100):
        # find the new velocity and locations and append them to __over_t
        locations, velocities = update(locations, velocities)

        pos_over_t.append(locations)
        vel_over_t.append(velocities)

    if dimensions == 2:
        # show paths in 2-D
        show_path_2D(pos_over_t, True)

    return 0


def pop_box():
    """
    Funciton which populates the box with N particles in random locations inside
    the box.
    """
    global L, dimensions, N

    # N must be less than L^2
    if N >=  L ** 2 / 2:
        print("N must be less than size of the box squared over 2")
        return 1

    # locations of the particles which will be created
    init_locations = []
    init_velocities = []

    count = 0
    # loop N times
    while count < N:
        # empty array  which will contain the location and velocities of the particle
        loc = []
        vel = []
        for i in range(dimensions):
            # create a random location in ith dimension
            loc.append(random.randint(0, L))
            vel.append(random.randrange(-100, 101, 100))

        # dont put it in the locations if a particle already exists
        if loc in init_locations:
            continue

        # put the particle in init_locations with its corresponding velocity
        init_locations.append(loc)
        init_velocities.append(vel)
        count += 1

    return init_locations, init_velocities


def update(positions, velocities):
    """
    Updates the position of an inputed coordinate based on the algortihtm ___.
    Returns the updated position.
    """
    global delta_t, dimensions, N

    # list which will contain new positions and velocities
    new_positions = []
    new_velocities = []

    # loop over each particle in the list
    for particle in range(N):

        # list which will contain new positions and velocities
        new_particle_position = []
        new_particle_velocity = []

        # loop over each dimension
        for i in range(dimensions):
            # update the new version of ith(x, y, z) position
            new_pos = positions[particle][i] + delta_t * velocities[particle][i]

            # update the version of velocity
            new_vel = velocities[particle][i] # for now

            # check new_pos for boundary conditions
            new_pos = preiodic_boundaries(new_pos)

            # put these values in new_pos, new_vel
            new_particle_position.append(new_pos)
            new_particle_velocity.append(new_vel)

        # put particle in new_positions array (and same for velocity)
        new_positions.append(new_particle_position)
        new_velocities.append(new_particle_velocity)

    return new_positions, new_velocities


def average_velocity(velocity_needing_update, velocities_close_particles):
    """
    Computes the average velocity in each dimension and outputs the average of this
    in each dimension.
    """
    global dimensions

    # list containing update velocities
    new_vel = []

    # loop araound each dimesnion and calculate the average in that dimension.
    for i in range(dimenisons):
        # initialise new velocity in this dimesnion to 0 to later get the mean
        new_vi = 0
        # loop over all particles in vicintiy
        for velocity in velocities_close_particles:
            # add the value of the velocity to previous
            new_vi += velocity[i]

        # divide by how many particles to get mean
        new_vi = new_vi / len(velocities_close_particles)

        # place value of new_vi in the new vel array
        new_vel.append(new_vi)

    return new_vel


def preiodic_boundaries(position):
    """
    If particle is over the limit of the box run this function and it will return
    the correct position of the particle.
    """
    global L

    # check if its under 0
    if position < 0:
        return position + L

    # if its over L
    elif position > L:
        return position - L

    # otherwise just return the position
    else:
        return position


def show_path_2D(coordinates, clear = False):
    """
    Function which takes in the coordinates as described in straight_particle and
    plots the result on a scatter graph.
    """
    global L, N, delta_t

    # start interactive mode
    plt.ion()

    # crete eempty figure on which data will go and first subplot
    fig = plt.figure()

    # get into the correct time step
    for time_step in coordinates:
        # list of colours used for animation
        colours = cm.rainbow(np.linspace(0, 1, N))

        # loop over each particle and colour
        for particle, c in zip(time_step, colours):
            # plot x, y poistion of particle in a given colour and set axis to size of box
            plt.scatter(particle[0], particle[1], s = 3, color = c)
            plt.axis([0, L, 0, L])

        # show graph
        plt.show()
        plt.pause(delta_t)

        # decide if you want to clear
        if clear == True:
            plt.clf()

    return None


main()
