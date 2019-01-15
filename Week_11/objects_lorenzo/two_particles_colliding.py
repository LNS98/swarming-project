"""
Program built to investiagte coding up the physics of object colliding.
"""


import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# constants used in the program
L = 10  # size of the box
N = 100  # number of particles
M = 10   # number of objects
v_mag = 0.05      # total magnitude of each particle velocity
delta_t = 1     # time increment
mass_par = 1 # masss of the particles
mass_object = 100 # masss of the particles

# distance metrics in the code
r = 1.0   # radius of allignment
r_c = 0.2 # radius within repulsion
r_e = 0.5 # radius of equilibrium between the particles
r_a = 0.8 # radius when attraction starts
r_o = 0.05 # radius of attraction between the particels and the objects

# force parrameters
alpha = 1 # stregnth of repulsive force due to the particles
beta = 10 # stregnth of the force due to the objects
gamma = 1 # stregnth of allignment

U = 1000    # number of updates
dimensions = 2   # dimensions
time_pause = 0.001 # time pause for interactive graph




def main():

    # fill up a box with particles and objects
    positions, velocities, accelerations = pop_box()
    positions_obj, velocities_obj, accelerations_obj = objects()

    # append the positions to the positions over time
    pos_part_over_t = [positions]
    vel_part_over_t = [velocities]
    pos_obj_over_t = [positions_obj]
    vel_obj_over_t = [velocities_obj]

    # update the position for 10 times
    for i in range(U):

        # call update to get the new positions of the particles
        positions, velocities = update_system(positions, velocities, accelerations, positions_obj)

        # update the positions of the objects
        positions_obj, velocities_obj = update_system_object(positions_obj, velocities_obj, accelerations_obj,
                                                                     positions, velocities)

        # append in positions over time
        pos_part_over_t.append(positions)
        vel_part_over_t.append(velocities)
        pos_obj_over_t.append(positions_obj)
        vel_obj_over_t.append(velocities_obj)

    # show the path of the particles
    show_path_2D(pos_part_over_t, pos_obj_over_t, clear = True)

    return 0

# ----------------------- System Functions ---------------------------------

def pop_box():
    """
    Function which creates one particle with a ramdom position and velocity
    in a square of dimensions L by L.
    """
    # will hold each posi/vel/acc for each particle in the system
    positions = []
    velocities = []
    accelerations = []

    for i in range(N):
        # lsit containing positions and velocities at random
        init_position = [random.uniform(0, L) for i in range(dimensions)]
        init_velocity = [random.uniform(-1, 1) for i in range(dimensions)]
        init_acceleration = [0 for i in range(dimensions)]

        # append the positions to the bigger lists
        positions.append(init_position)
        velocities.append(rescale(v_mag, init_velocity))
        accelerations.append(init_acceleration)

    return positions, velocities, accelerations

def objects():
    """
    Create a set of M objects, defining them just by there  centre of mass.
    As of now they are basically particles of different species.
    """

    # will hold each posi/vel/acc for each particle in the system
    positions = []
    velocities = []
    accelerations = []

    for i in range(M):
        # lsit containing positions and velocities at random
        init_position = [random.uniform(0, L) for i in range(dimensions)]
        init_velocity = [0 for i in range(dimensions)]
        init_acceleration = [0 for i in range(dimensions)]

        # append the positions to the bigger lists
        positions.append(init_position)
        velocities.append(init_velocity)
        accelerations.append(init_acceleration)

    return positions, velocities, accelerations

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

# ----------------------- Update Functions for particles --------------------

def update_system(positions, velocities, accelerations, positions_obj):
    """
    Updates the positons and velocities of ALL the particles in a system.
    """
    # lists which will contain the updated values
    new_positions = []
    new_vels = []

    # loop through each index in the positions, vel, acc
    for i in range(N):
        # get the acceleration based on the positions of the particles
        acceleration = update_acceleration(positions[i], velocities[i], positions, velocities, positions_obj)
        # call update to get the new value
        new_vel = update_velocity(velocities[i], acceleration)
        new_pos = update_position(positions[i], new_vel)
        # append it to the new values
        new_positions.append(new_pos)
        new_vels.append(new_vel)

    return new_positions, new_vels

def update_position(position, velocity):
    """
    Update the location of a particle and returns the new location.
    """
    # create a new lsit which will contain the new position
    new_pos = []

    # loop through the dimensions in position
    for i in range(dimensions):

        # add the velocity in that dimension to the position (times delta_t)
        pos_i = position[i] + velocity[i] * delta_t
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

    return new_vel

def update_acceleration(position_particle, velocity_particle, position_particles, velocity_particles, positions_obj):
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
        force_particles += chate_rep_att_force(position_particle, particle)

    # calcualte force due to the objects
    for object in positions_obj:
        force_object += obj_repulsive_force(position_particle, object)


    new_acceleration = (alpha * force_particles + beta * force_object +
    gamma * allignment_force(position_particle, velocity_particle, position_particles, velocity_particles)) / mass_par

    return new_acceleration

# ----------------------- Update Functions for objects ------------------------------

def update_system_object(positions_obj, velocities_obj, accelerations_obj, position_particles, velocity_particles):
    """
    Updates the positons and velocities of ALL the particles in a system.
    """
    # lists which will contain the updated values
    new_positions = []
    new_vels = []

    # loop through each index in the positions, vel, acc
    for i in range(len(positions_obj)):
        # get the acceleration based on the positions of the particles
        acceleration = update_acceleration_object(positions_obj[i], positions_obj, position_particles, velocity_particles)
        # call update to get the new value
        new_vel = update_velocity_object(velocities_obj[i], acceleration)
        new_pos = update_position_object(positions_obj[i], new_vel)
        # append it to the new values
        new_positions.append(new_pos)
        new_vels.append(new_vel)

    return new_positions, new_vels

def update_position_object(positions_obj, velocities_obj):
    """
    Update the location of a particle and returns the new location.
    """
    # create a new lsit which will contain the new position
    new_pos = []

    # loop through the dimensions in position
    for i in range(dimensions):

        # add the velocity in that dimension to the position (times delta_t)
        pos_i = positions_obj[i] + velocities_obj[i] * delta_t
        # print(positions_obj[i], velocities_obj[i], pos_i)
        pos_i = periodic_boundaries(pos_i)

        # append to the new_position and velocity list this position/velocity
        new_pos.append(pos_i)

    return new_pos

def update_velocity_object(velocities_obj, accelerations_obj):
    """
    Update the velocity of a particle and returns the new velocity.
    """
    # create a new lsit which will contain the new position
    new_vel = []

    # loop through the dimensions in position
    for i in range(dimensions):
        # update the velocity first
        v_i = velocities_obj[i] + accelerations_obj[i] * delta_t

        # append to the new_position and velocity list this position/velocity
        new_vel.append(v_i)

    return new_vel

def update_acceleration_object(position_obj, positions_obj, position_particles, velocity_particles):
    """
    Algorithm which updates the algorithm
    """
    # define two inital forces dependent on the particles and on hte object
    force_object = np.array([0., 0.])
    force_particles = np.array([0., 0.])

    # loop through each particle and calculate the repulsive force from the particle
    for particle in position_particles:
        force_particles += obj_repulsive_force(position_obj, particle)

    # calcualte force due to the objects
    for object in positions_obj:
        if object == position_obj:
            continue
        force_object += obj_repulsive_force(position_obj, object)

    new_acceleration = (beta * force_particles + alpha * force_object) / mass_object

    return new_acceleration

# ----------------------- Forces Functions ------------------------------

def obj_repulsive_force(i, j):
    """
    calculates the force used in the repulsive_force function.
    """
    # calculate the distance between the points
    distance_x, distance_y = per_boun_distance(i, j)

    # calcualte the magnitude of the distance between the points
    distance = (distance_x ** 2 + distance_y ** 2) ** (1/2)

    # magnitude of force
    magnitude = -1 /(1 + math.exp(distance/ r_o))

    # get the x direction of the force
    F_x = (magnitude * distance_x) / distance

    # get the y direction of the force
    F_y = (magnitude * distance_y) / distance

    return np.array([F_x, F_y])

def inverse_force(i, j):
    """
    (1/r)^2 repulsive force
    """
    # calculate the distance between the points
    distance_x, distance_y = per_boun_distance(i, j)

    # calcualte the magnitude of the distance between the points
    distance = (distance_x ** 2 + distance_y ** 2) ** (1/2)

    # magnitude of force
    magnitude = - (1/distance) ** 2

    # get the x direction of the force
    F_x = (magnitude * distance_x) / distance

    # get the y direction of the force
    F_y = (magnitude * distance_y) / distance

    return np.array([F_x, F_y])

def allignment_force(position_particle, velocity_particle, position_particles, velocities_particles):
    """
    Add a force which changes the velocity in the direction of the desired one.
    """
    # convert the velocities to numpy arrays
    velocity_particle = np.array(velocity_particle)

    # get the velocity of particles in radius
    vel_in_r = np.array(particles_in_radius(position_particle, position_particles, velocities_particles)[0])

    # get the average value of that velocity
    vel_wanted = np.mean(vel_in_r, axis = 0)

    # get the force by subtracting the current vel from the desirerd one
    Force = vel_wanted - velocity_particle

    return Force

def chate_rep_att_force(i, j):
    """
    Attractive and repulsive force between the particles as described in the
    chate paper 2003.
    """

    # calculate the distance between the points
    distance_x, distance_y = per_boun_distance(i, j)

    # calcualte the magnitude of the distance between the points
    distance = (distance_x ** 2 + distance_y ** 2) ** (1/2)

    # if distance smaller than r_c
    if distance < r_c:
        # basically inifinite force
        magnitude = 1e10

    # if distnace between r_c and r_a (the radius of attraction)
    if r_c < distance < r_a:
        # force towards r_e (the equilibrium distance)
        magnitude = (1/4) * (distance - r_e) / (r_a - r_e)

    # if beyond ra but smaller than r_0
    if r_a < distance < r:
        # magnitude attraction
        magnitude = 1

    # else no force
    else:
        magnitude = 0

    # get the x direction of the force
    F_x = (magnitude * distance_x) / distance

    # get the y direction of the force
    F_y = (magnitude * distance_y) / distance

    return np.array([F_x, F_y])

# ----------------------- Visualise Functions ------------------------------

def show_path_2D(coordinates, coordinates_object, clear = True):
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
    for time_step in range(len(coordinates)):
        # list of colours used for animation
        colours = cm.rainbow(np.linspace(0, 1, N))

        # loop over each particle and colour
        for i in range(N):
            # plot x, y poistion of particle in a given colour and set axis to size of box
            plt.scatter(coordinates[time_step][i][0], coordinates[time_step][i][1], s = 3, color = 'r')

            # plot the object
            if i < M:
                # plt.plot(verticies[0] , verticies[1])
                plt.scatter(coordinates_object[time_step][i][0], coordinates_object[time_step][i][1], s = 8, color = 'g')

            plt.axis([0, L, 0, L])

        # show graph
        plt.show()
        plt.pause(time_pause)

        # decide if you want to clear
        if clear == True:
            plt.clf()

    return None

# ----------------------- Help Functions ------------------------------

def particles_in_radius(position_particle, position_particles, velocities_particles):
    """
    Checks and records the particles which are within radius r.
    Returns the veloicities and positions of those particles that
    are within radius r.
    """

    # array with all indecies of all particles within range for velocities
    velocities_within_r = []

    # array with all indecies of all particles within range for positions
    positions_within_r = []

    # check over all particles in positions
    for index in range(N):
        # variable used to aid if its in radius
        in_size = True

        # check if it is smaller than the radius in all
        for i in range(dimensions):

            inside_distance = abs(position_particle[i] - position_particles[index][i])

            wrap_distance = L-inside_distance

            distance = min(inside_distance, wrap_distance)

            # if the size is over then break out of loop as it won't be in radius
            if distance > r:
                in_size = False
                break

        # If it is within radius, add velocity to all velociites within r
        if in_size == True:
            # get the index of the particle for velocity
            velocities_within_r.append(velocities_particles[index])

            # get the index of the particle for position
            # and add position to all positions within r
            positions_within_r.append(position_particles[index])


    return velocities_within_r, positions_within_r

def angle_to_xy(angle):
    """
    Takes an angle in radians as input as returns x and y poistion for the corresponding angle
    using r as v_mag, a predifined value which is the magnitude of the velocity.
    """
    # get x using x = cos(angle) and y = sin(angle)
    x = v_mag * math.cos(angle)
    y = v_mag * math.sin(angle)

    return x, y

def rescale(magnitude, vector):
    """
    Changes the length of a  given vector to that of the magnitude given.
    """
    # make the vector a numpy array
    vec = np.array(vector)

    # get the magnitude
    mag = np.sqrt(vec.dot(vec))

    # multiply to rescale and make it a list
    new_vec = (magnitude / mag) * vec
    new_vec = list(new_vec)

    return new_vec

def per_boun_distance(i, j):
    """
    Calculates the minimum distance  between two particles in a box with periodic
    boundries.
    """
    # calculate the minimum x distance
    in_distance_x = j[0] - i[0]
    out_distance_x = L - in_distance_x
    distance_x = min(in_distance_x, out_distance_x)


    # calculate the minimum y distance
    in_distance_y = j[1] - i[1]
    out_distance_y = L - in_distance_y
    distance_y = min(in_distance_y, out_distance_y)

    return distance_x, distance_y

def help():
    """
    Funciton used for different reasons.
    """
    a = np.array([3, 4])
    b = np.array([2, 3])

    c = 5 * a
    print(c)

    return None


# run program
start = time.time()
main()
# help()
print("------------------------- Time Taken: {} -------------------".format(time.time() - start))
