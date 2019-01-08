"""
Get lots of particles on the thing and make them allign.
However, code is buggy and the particles produces random walks 
so the alignment function needs to be altered. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# global variables used in the program
L = 10     # size of the box
delta_t = 1     # time increment
dimensions = 2   # dimensions
N = 10   # number of particles
U = 1000    # number of updates

def main():
    """
    Execution of main program.
    """

    # all positions over time
    pos_over_t = []
    # all velocities over time
    vel_over_t = []

    # populate the box - returns initial poss and vels
    positions, velocities = pop_box()

    # add init vel and poss to pos/vel over time
    pos_over_t.append(positions)
    vel_over_t.append(velocities)

    # update position of each particle in the box
    for i in range(U):

        # update velocities dependant on previous positions
        velocities = update_vel(positions, velocities)

        # add new velocities to array over time
        vel_over_t.append(velocities)

        # update positions dependant on previous velocities
        positions = update_pos(velocities, positions)

        # add new positions in array over time
        pos_over_t.append(positions)

    if dimensions == 2:
        # show paths in 2-D
        show_path_2D(pos_over_t, clear = True)

    return 0

# returns initial positions and velocities of all particles as array
def pop_box():
    """
    Funciton which populates the box with N particles in random
    locations and random velocities inside the box.
    """
    global L, dimensions, N

    # N must be less than L^2
    if N >=  L ** 2 / 2:
        print("N must be less than size of the box squared over 2")
        return 1

    # initial positions of created particles
    init_positions = []
    # initial velocities of created particles
    init_velocities = []

    count = 0
    # loop N times
    while count < N:
        # this will contain the positions and velocities of single particle (x, y, z)
        pos = []
        vel = []

        for i in range(dimensions):
            # create a random position for each dimension
            pos.append(random.randint(0, L))
            vel.append(random.randint(-100, 100) / 100)

        # don't put it in the positions if a particle already exists there
        if pos in init_positions:
            continue

        # put the particle in init_positions with its corresponding velocity
        init_positions.append(pos)
        init_velocities.append(vel)
        count += 1

    #returns the initial positions and velocities of all particles
    return init_positions, init_velocities

# returns array of updated positions
def update_pos(positions, velocities):
    """
    Updates locations
    """
    # array to contain updated positions of all particles
    new_positions = []

    for particle in range(N):

        # array with updated coordinates of each particle(x, y, z)
        new_particle_position = []

        # update position in each dimension
        for i in range(dimensions):
            new_pos = positions[particle][i] + delta_t * velocities[particle][i]

            #check periodic boundary conditions
            new_pos = periodic_boundaries(new_pos)

            new_particle_position.append(new_pos)

        # append each particle to array of updated particle positions
        new_positions.append(new_particle_position)

    return new_positions


# returns array of updated velocities
def update_vel(positions, velocities):
    """
    Updates velocity
    """
    global delta_t, dimensions, N

    #list will contain new velocities of all particles
    new_velocities = []

    #loop over each particle
    for particle in range(N):

        # find all the close particles
        close_particles_vel = particles_in_sq(velocities[particle], positions, velocities)

        #update velocity of each particle
       #function already loops through dimentions
        new_vel = new_vel_of_particle(velocities[particle], close_particles_vel)

        #new velocity arrays
        new_velocities.append(new_vel)

    #returns new velocities of all particles
    return new_velocities


# returns new velocity of a single particle in V coord
def new_vel_of_particle(velocity, close_particles_velocities):
    """
    Computes the average velocity in each dimension and outputs the average of this
    in each dimension.
    """
    global dimensions

    # if no particle close to it  return the same velocity
    if len(close_particles_velocities) == 0:
        return [random.randint(-100, 100) / 100, random.randint(-100, 100) / 100]

    # list containing updated velocities Vx, Vy, Vz
    new_vel = []

    # all particles array
    # all_vicinity_velocities = particles_in_sq(velocities[particle])

    # loop araound each dimesnion and calculate the average in that dimension.
    for i in range(dimensions):
        # initialise new velocity in this dimesnion to 0 to later get the mean
        new_vi = 0

        # loop over all particles in vicintiy
        for velocity in close_particles_velocities:
            # add the value of the velocity to previous
            new_vi += velocity[i]

        # divide by how many particles to get mean
        new_vi = new_vi / len(close_particles_velocities)

        # place value of new_vi in the new vel array
        new_vel.append(new_vi)

    return new_vel

#This function is wrong FIX IT....
def particles_in_sq(chosen_particle, positions, velocities):
    """
    Checks and records the particles which are within a square of lengt r.
    """
    # radius
    r = L * 0.1

    # array with all indecies of all particles within range
    velocities_within_r = []

    # check over all particles in positions
    for index in range(N):
        # variable used to aid if its in radius
        in_size = True

        # skip particle if its the chosen one
        if positions[index] == chosen_particle:
            continue

        # check if it is smaller than the radius in all
        for i in range(dimensions):

            inside_distance = abs(chosen_particle[i] - positions[index][i])
            wrap_distance = L-inside_distance

            distance = min(inside_distance, wrap_distance)

            # if the size is over then break out of loop as it won't be in radius
            if distance > r:
                in_size = False
                break

        #If it is within square add velocity to all particles within r
        if in_size == True:
            # get the index of the particle
            velocities_within_r.append(velocities[index])

    return velocities_within_r


def test_in_sq():
    """
    Test function for particles_in_radius.
    """

    # get some positions for x and y locations of N particles
    x_pos = [random.randint(0, L) for i in  range(N)]
    y_pos = [random.randint(0, L) for i in  range(N)]

    # positions of particels as [x, y]
    positions = [[x_pos[i], y_pos[i]] for i in range(N)]

    print(positions)
    print(positions[0])

    # plot the graph to see visually what happens
    # plt.scatter(x_pos, y_pos)
    # plt.show()

    # call in radius function to see whether it works correctly
    close_1 = particles_in_sq(positions[0], positions)
    print(close_1)

def periodic_boundaries(position):
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

def show_path_2D(coordinates, clear = True):
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
        plt.pause(0.05)

        # decide if you want to clear
        if clear == True:
            plt.clf()

    return None


main()
