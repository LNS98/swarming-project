"""
Vicsek Model with alignment vs noise graph produced - highlighting the phase transition between the two states
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import random

# global variables used in the program
L = 10  # size of the box
delta_t = 1     # time increment
v_mag = 0.03      # total magnitude of each particle velocity
dimensions = 2   # dimensions
N = 100 # number of particles
r = 0.5  #radius
U = 100    # number of updates
noise = 0 # magnitude of varied noise
time_pause = 0.001 # time pause for interactive graph

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

    # print allignment at start
    print("Allignment at start is: {}".format(allignment(velocities)))

    # add init vel and poss to pos/vel over time
    pos_over_t.append(positions)
    vel_over_t.append(velocities)


    #start array for time
    time = []
    # start array for allignment at each timestep
    allignment_array = []


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


        # append count to time array to keep track of timestep
        time.append(i)
        # append allignment to array of allignments at each time
        allignment_array.append(allignment(velocities))


    # print allignment at end
    print("Allignment at end is: {}".format(allignment(velocities)))

    # plot allignment vs time to see how it depends on noise
    # show_allignment_plot(time, allignment_array)

    if dimensions == 2:
        # show paths in 2-D
        show_path_2D(pos_over_t, clear = True)

    return 0


def angle_to_xy(angle):
    """
    Takes an angle in radians as input as returns x and y poistion for the corresponding angle
    using r as v_mag, a predifined value which is the magnitude of the velocity.
    """
    # get x using x = cos(angle) and y = sin(angle)
    x = v_mag * math.cos(angle)
    y = v_mag * math.sin(angle)

    return x, y

def test_angle_form():
    """
    Test formula for angle_to_xy.
    """
    # populate angles
    angles = []
    angle = - math.pi
    for i in range(5):
        angles.append(angle)
        angle += 2 * math.pi / 4

    # run function for angles
    for angle in angles:
        x, y = angle_to_xy(angle)
        print(x, y)

    return None


def pop_box():
    """
    Funciton which populates the box with N particles in random
    locations and random velocities inside the box.
    """
    global L, dimensions, N

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

        # get a random angle in radians to use as the velocity of the particle
        angle = random.uniform(-math.pi, math.pi)

        for i in range(dimensions):
            # create a random position for each dimension
            pos.append(random.uniform(0, L))
            vel.append(angle_to_xy(angle)[i])

       # don't put it in the positions if a particle already exists there
        if pos in init_positions:
            continue

        # put the particle in init_positions with its corresponding velocity
        init_positions.append(pos)
        init_velocities.append(vel)
        count += 1

    #returns the initial positions and velocities of all particles
    return init_positions, init_velocities


def update_pos(positions, velocities):
    """
    Updates locations dependign on current velocity
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


def update_vel(positions, velocities):
    """
    Updates velocity
    """
    global delta_t, dimensions, N

    #list will contain new velocities of all particles
    new_velocities = []

    #loop over each particle
    for particle in range(N):

#        print("\nparticle in question: {}".format(positions[particle]))

        # find all the close particles
        close_particles_vel = particles_in_sq(positions[particle], positions, velocities)

#        print("positons of particles: {} ".format(positions))
#        print("velocities of particles: {} ".format(velocities))

#        print("velocities of close particles: {}".format(close_particles_vel))
        #update velocity of each particle
        #function already loops through dimentions
        new_vel = new_vel_of_particle(velocities[particle], close_particles_vel)

#        print("new particle velocity: {}".format(new_vel))

        #new velocity arrays
        new_velocities.append(new_vel)

    #returns new velocities of all particles
    return new_velocities


def new_vel_of_particle(velocity, close_particles_velocities):
    """
    Computes the average velocity in each dimension and outputs the average of this
    in each dimension.
    """
    global dimensions

    # list containing updated velocity Vx, Vy, Vz
    new_vel = []

    # loop araound each dimesnion and calculate the average in that dimension.
    for i in range(dimensions):
        # initialise new velocity in this dimesnion to 0 to later get the mean
        new_vi = 0

        # loop over all particles in vicintiy
        for velocity in close_particles_velocities:
            # add the value of the velocity to previous
            new_vi += velocity[i]

        # divide by how many particles to get mean
        average_vi = (new_vi) / len(close_particles_velocities)

        # place value of new_vi in the new vel array
        new_vel.append(average_vi)

    # get the new direction by applying pheta = arctan(<x> / <y>)
    if new_vel[0] == 0:
        # if y = 0 then make it go 90 deg
        angle = math.pi / 4 + random.uniform(- noise / 2, noise / 2)
    else:
        # get the new direction
        angle = math.atan2(new_vel[1], new_vel[0]) + random.uniform(- noise / 2, noise / 2)

    # convert direction to x, y coordinates.
    new_vel[0], new_vel[1] = angle_to_xy(angle)

    return new_vel


def particles_in_sq(chosen_particle, positions, velocities):
    """
    Checks and records the particles which are within a square of lengt r.
    """

    # array with all indecies of all particles within range
    velocities_within_r = []

    # check over all particles in positions
    for index in range(N):
        # variable used to aid if its in radius
        in_size = True

#        print("particles being compared: {} {}".format(chosen_particle, positions[index]))
        # check if it is smaller than the radius in all
        for i in range(dimensions):

            inside_distance = abs(chosen_particle[i] - positions[index][i])

            wrap_distance = L-inside_distance

            distance = min(inside_distance, wrap_distance)

            # if the size is over then break out of loop as it won't be in radius
            if distance > r:
#                print("particle distance and 'r' (inside sq function): {} and {}".format(distance, r))
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

    x_vel = [random.randint(0, L) for i in  range(N)]
    y_vel = [random.randint(0, L) for i in  range(N)]

    # positions of particels as [x, y]
    positions = [[x_pos[i], y_pos[i]] for i in range(N)]
    velocities = [[x_vel[i], y_vel[i]] for i in range(N)]

    print(positions)
    print(velocities)

    # call in radius function to see whether it works correctly
    close_1 = particles_in_sq(positions[0], positions, velocities)

    print(close_1)

    # plot the graph to see visually what happens
    plt.scatter(x_pos, y_pos)
    plt.axis([0, L, 0, L])
    plt.show()


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


def allignment(velocities):
    """
    Calculates the net allignment of the velocities of all the particles and
    normmailses this value so that if they are all alligned the allignment is 1.
    """
    # initialise values for sum of all vx and vy
    vx = 0
    vy = 0

    # sum all velocities in velocities array
    for particle in velocities:
        #add vx particle to sum of all vx
        vx += particle[0]

        #add vy particle to sum of all vy
        vy += particle[1]

    # Total magnitude of velocity of particles
    v_mag_tot = math.sqrt(vx**2 + vy**2)

    # Check alignment of particles
    v_a = (1/(N * v_mag)) * (v_mag_tot)

    return v_a


def show_allignment_plot(time, allignment):

    # plot time vs allignment for x vs y
    plt.plot(time, allignment, linewidth=1, marker=".", markersize=3)
    plt.xlabel("Time")
    plt.ylabel("Allignment value")
    plt.show()

    return None

# def show_allignment_interactive(time, allignment):
#     """
#     Function which will show how allignment changes as time evlolves
#     """
#
#     # start interactive mode
#     plot.ion()
#
#     # empty figure on which data will go
#     fig = plt.figure()


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
            # plt.plot([particle[0] - r, particle[0] + r, particle[0] + r, particle[0] - r, particle[0] - r],
            #          [particle[1] - r, particle[1] - r, particle[1] + r, particle[1] + r, particle[1] - r],
            #          color = c)
            plt.axis([0, L, 0, L])

        # show graph
        plt.show()
        plt.pause(time_pause)

        # decide if you want to clear
        if clear == True:
            plt.clf()

    return None


main()
