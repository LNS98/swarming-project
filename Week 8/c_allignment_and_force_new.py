"""
This is an unfinished program of the incorporation of alignment and repulsive force.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import random

# global variables used in the program - same as Vicsek Model, not needed things are commented out
L = 31.6  # size of the box
delta_t = 1     # time increment
v_mag = 0.003      # total magnitude of each particle velocity
dimensions = 2   # dimensions
N = 20 # number of particles
r = 2 # radius for vicsek model
a = 10 #radius for Marchetti model will be called a in this program
U = 1000   # number of updates
noise_vic = 0 # magnitude of varied noise for vicshek model
noise_mar = 0 # magnitude of varied noise for marchetti model
time_pause = 0.001 # time pause for interactive graph
k = 0.1 # spring constant ???
mu = 0.1 # ??? mobility
V = 0 #strenght of viscek
M = 0.5 # strnehgt of marchetti


def main():
    """
    Execution of main program.
    """
    global noise_vic

    # all positions over time
    pos_over_t = []
    # all velocities over time
    vel_over_t = []

    # populate the box - returns initial poss and vels
    positions, velocities = pop_box()

    # add init vel and poss to pos/vel over time
    pos_over_t.append(positions)
    vel_over_t.append(velocities)


    #start array for time
    time = []



    # update position of each particle in the box
    for i in range(U):
        # start turning up the noise once the particles have come to a staionary state and dont let the noise go over 5
        if i > 300 and i % 100 == 0:
            noise_vic += 0.5

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


    if dimensions == 2:
        # show paths in 2-D
        show_path_2D(pos_over_t, clear = True)

    return 0


def pop_box():
    """
    Function which populates the box with N particles in random
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
            vel.append(angle_to_xy(v_mag, angle)[i])

       # don't put it in the positions if a particle already exists there
        if pos in init_positions:
            continue

        # put the particle in init_positions with its corresponding velocity
        init_positions.append(pos)
        init_velocities.append(vel)
        count += 1

    #returns the initial positions and velocities of all particles
    return init_positions, init_velocities

def angle_to_xy(r, angle):
    """
    Takes an angle in radians as input as returns x and y poistion for the corresponding angle
    using r as v_mag, a predifined value which is the magnitude of the velocity.
    """
    # get x using x = cos(angle) and y = sin(angle)
    x = r * math.cos(angle)
    y = r * math.sin(angle)

    return x, y

def test_xy_angle():
    """
    test to see if inverting between a angle and vice e versa works as expected
    """
    # angle = math.atan2(new_vel[1], new_vel[0])

    # random x, y
    x, y = -0.02597136287429149, -0.14727925296049865
    r = mag([x, y])

    # get the angle for the x, y positions
    angle = math.atan2(y, x)
    print(angle)

    # get the x, positions back
    x, y = angle_to_xy(r, angle)
    print(x, y)

    return None


def mag(vector):
    """
    Finds the magnitude of a vector.
    """
    return math.sqrt(sum(i**2 for i in vector))


def update_pos(positions, velocities):
    """
    Updates locations depending on current velocities and returns new positions
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
    Updates velocity based on forces with neighbours
    """
    global delta_t, dimensions, N

    # list will contain new velocities of all particles
    new_velocities = []

    # loop over each particle
    for particle in range(N):

        # find all the close particles positions and velocities for marchetti model
        close_particles_vel_mar, close_particles_pos_mar = particles_in_sq(a, positions[particle], positions, velocities)
        # find all close particles for vicsek model
        close_particles_vel_vic = particles_in_sq(r, positions[particle], positions, velocities)[0]

        # update velocity of each particle
        # function already loops through dimentions
        # update for new velocity given vicshek model
        new_vel_vic = new_vel_of_particle_vic(velocities[particle], close_particles_vel_vic)

        # update new velocity for marchetti model
        new_vel_mar = new_vel_of_particle_mar(velocities[particle], positions[particle], close_particles_vel_mar, close_particles_pos_mar)

        print(new_vel_mar[0], new_vel_mar[1])

        new_vel_tot = [(V*new_vel_vic[0] + M*(new_vel_mar[0])), (V*new_vel_vic[1] + M*(new_vel_mar[1]))]


        # append the new velocity of each particle to array of all new vels
        new_velocities.append(new_vel_tot)

    #returns array of new velocities for all particles
    return new_velocities

def new_vel_of_particle_vic(velocity, close_particles_velocities):
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
        for velo in close_particles_velocities:
            # add the value of the velocity to previous
            new_vi += velo[i]

        # divide by how many particles to get mean
        average_vi = (new_vi) / len(close_particles_velocities)

        # place value of new_vi in the new vel array
        new_vel.append(average_vi)



    # get the new direction by applying pheta = arctan(<x> / <y>)
    if new_vel[0] == 0:
        # if y = 0 then make it go 90 deg
        angle = math.pi / 4 + random.uniform(- noise_vic / 2, noise_vic / 2)
    else:
        # get the new direction
        angle = math.atan2(new_vel[1], new_vel[0]) + random.uniform(- noise_vic / 2, noise_vic / 2)

    # convert direction to x, y coordinates.
    new_vel[0], new_vel[1] = angle_to_xy(v_mag, angle)


    return new_vel

def new_vel_of_particle_mar(velocity, position, close_particles_velocities, close_particles_positions):
    """
    Computes the average velocity in each dimension according to all and outputs the average of this
    in each dimension, also includes rotational noise
    considers force
    considers partciles only within radius 2r
    """
    global dimensions

    # list containing updated velocity Vx, Vy
    new_vel = []

    # loop araound each dimesnion and calculate the average in that dimension.
    for i in range(dimensions):

        # Initialisation of each dimention of vel
        vel_coord = 0

        # loop over all particles in vicinity
        for j in close_particles_positions:
            # skip same particle, i
            if j == position:
                continue

            # Calc force in current dimension
            force_coord = (force_mag(j, position)*(j[i]-position[i]))/distance(j, position)

            # add force to current vel
            vel_coord += force_coord

        # multiply by mu
        vel_coord *= mu

        # add v0
        vel_coord += v_mag

        # place value of each new velocity for x/y in array for new total vel
        new_vel.append(vel_coord)

    # get the new direction by applying pheta = arctan(<x> / <y>)
    if new_vel[0] == 0:
        # if y = 0 then make it go 90 deg
        angle = math.pi / 4 + random.uniform(- noise_mar / 2, noise_mar / 2)
    else:
        # get the new direction
        angle = math.atan2(new_vel[1], new_vel[0]) + random.uniform(- noise_mar / 2, noise_mar / 2)

    new_vel[0], new_vel[1] = angle_to_xy(mag(new_vel), angle)


# Noise formula DO NOT DELETE random.uniform(- noise / 2, noise / 2)

    # convert direction to x, y coordinates.
    # new_vel[0], new_vel[1] = angle_to_xy(angle)

    return new_vel

def force_mag(part_i, part_j):
    '''
    Calculates force between two particles
    '''

    Fij = -k * ( 2*a - distance(part_i, part_j))
    return Fij

def distance(p_1_pos, p_2_pos):
    '''
    Calculates distance between two particles at certain positions
    '''
    XY = []
    for i in range(dimensions):

        dist = abs(p_1_pos[i]-p_2_pos[i])

        wrap_dist = L-dist

        real_dist = min(wrap_dist, dist)

        XY.append(real_dist)

    rij = math.sqrt(XY[0]**2 + XY[1]**2)

    return rij

def particles_in_sq(a, chosen_particle, positions, velocities):
    """
    Checks and records the particles which are within radius.
    Returns only velocities which are within radius.
    """

    # arrays with all indecies of all particles within range
    velocities_within_a = []
    positions_within_a = []

    # check over all particles in positions
    for index in range(N):
        # variable used to aid if its in radius
        in_size = True

        # check if it is smaller than the radius in all
        for i in range(dimensions):

            inside_distance = abs(chosen_particle[i] - positions[index][i])

            wrap_distance = L-inside_distance

            distance = min(inside_distance, wrap_distance)

            # if the size is over then break out of loop as it won't be in radius
            if distance > a:
                in_size = False
                break

        #If it is within square add velocity to all particles within r
        if in_size == True:
            # get the index of the particle
            velocities_within_a.append(velocities[index])
            positions_within_a.append(positions[index])


    return velocities_within_a, positions_within_a

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
            plt.scatter(particle[0], particle[1], s = 15, color = c)
            # plt.plot([particle[0] - r, particle[0] + r, particle[0] + r, particle[0] - r, particle[0] - r],
            #          [particle[1] - r, particle[1] - r, particle[1] + r, particle[1] + r, particle[1] - r],
            #          color = c)
            plt.axis([0, L, 0, L])
            plt.title("a = {}, r = {}\nnoise vischek = {}, noise marchetti = {}".format(a, r, noise_vic, noise_mar))


        # show graph
        plt.show()
        plt.pause(time_pause)

        # decide if you want to clear
        if clear == True:
            plt.clf()

    return None

main()
