"""
Program built to investiagte coding up the physics of object colliding.
"""


import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# constants used in the program
L = 3.1  # size of the box
N = 40  # number of particles
M = 0   # number of objects
v_mag = 0.05      # total magnitude of each particle velocity
delta_t = 1     # time increment
mass_par = 1 # masss of the particles
mass_object = 100 # masss of the particles
noise = 0.5  # noise added to the acceleration

# distance metrics in the code
r = 1.0   # radius of allignment
r_c = 0.2 # radius within repulsion
r_e = 0.5 # radius of equilibrium between the particles
r_a = 0.8 # radius when attraction starts
r_o = 0.05 # radius of attraction between the particels and the objects
k = 7 # number of nearest neighbours

# force parameters
alpha = 0 # stregnth of repulsive force due to the particles
beta = 0 # stregnth of the force due to the objects
gamma = 1 # stregnth of allignment

# picking a model
model = "SVM" # select SVM for standard Vicsek Model and kNN for nearest neighbours

U = 800   # number of updates
dimensions = 2   # dimensions
time_pause = 0.001 # time pause for interactive graph



def main():

    one_run(plot2 = True)

    return 0

# ----------------------- Whole system Functions ---------------------------------

def one_run(plot1 = False, plot2 = False):
    """
    One simulation of a total run by the system.
    """

    # fill up a box with particles and objects
    positions, velocities, accelerations = pop_box()
    positions_obj, velocities_obj, accelerations_obj = objects()

    # append the positions to the positions over time
    pos_part_over_t = [positions]
    vel_part_over_t = [velocities]
    pos_obj_over_t = [positions_obj]
    vel_obj_over_t = [velocities_obj]

    align_start = allignment(velocities)

    # list for average nearest neighbours at each update U
    averages_list = []
    for_total_average = []
    U_list = []

    # update the position for 10 times
    for i in range(U):

        # call update to get the new positions of the particles
        positions, velocities, average = update_system(positions, velocities, accelerations, positions_obj)

        # update the positions of the objects
        positions_obj, velocities_obj = update_system_object(positions_obj, velocities_obj, accelerations_obj,
                                                                     positions, velocities)

        # append in positions over time
        pos_part_over_t.append(positions)
        vel_part_over_t.append(velocities)
        pos_obj_over_t.append(positions_obj)
        vel_obj_over_t.append(velocities_obj)

        averages_list.append(average)
        U_list.append(i)

        if U > 60: for_total_average.append(average)

    align_end = allignment(velocities)

    # plot the movment of the particles if plot is set to true
    if plot1 == True:
        show_path_2D(U - U, U, pos_part_over_t, pos_obj_over_t, clear = True)

    if plot2 == True:
        # plot average neighbours for each time step
        plot_average_neighbours(averages_list, U_list)

    np_average = np.array(for_total_average)
    mean = np.mean(np_average)

    return align_end, mean

def variation(type):
    """
    calcualtes the allignment of the systems for different values of the noise/density.
    """

    # create a list containg the values of noise tested
    noise_list = list(np.linspace(0, 5, num = 20))
    density_list = list(np.linspace(0.0001, 3, num = 15)) + list(np.linspace(3.5, 10, num = 5))
    # L_list = [3.1, 5, 10, 31.6, 50]
    # N = [40, 100, 400, 4000, 10000]

    neighbours_list = []

    if type == "noise":

        # for each value in list run main funciton
        for no in noise_list:
            # change the noise to new value of noise for the global variable noise
            global noise
            noise = no

            # print("no")
            # print(no)
            # get the NN from the main funciton
            all = one_run()[1]

            # print("average NN for each no")
            # print(all)

            # append this to the all_list
            neighbours_list.append(all)

        # print("neighbours_list:")
        # print(neighbours_list)

        return noise_list, neighbours_list

    if type == "density":
        # for each value in list run main funciton
        for density in density_list:
            # change the noise to new value of noise for the global variable noise
            global L
            L = (N / density) ** (1 / 2)
            # get the allignnment from the main funciton
            all = one_run()[1]

            # append this to the all_list
            neighbours_list.append(all)

        return density_list, neighbours_list
    else:
        print("not the correct 'type' given, try 'noise' or 'density'.")
        return None

def average_noise_allignment(n_times, type):
    """
    Create a file of ongoing repeats for the given 'type' of average.
    """

    # list with values of nosie, L and N
    noise_list = list(np.linspace(0, 5, num = 20))
    density_list = list(np.linspace(0.0001, 3, num = 15)) + list(np.linspace(3.5, 10, num = 5))

    if type == "noise":
        corr_list = noise_list
    if type == "density":
        corr_list = density_list

    # for each repeat, add new allignment values to old allignment values
    for repeat in range(n_times):

        # read in data from file if file exists
        try:
            # read in csv as dataframe
            df = pd.read_csv("./averages_{}_{}/N_{}.csv".format(type, model, N))
            average_number = len(df.columns) - 1

        except IOError as e:
            # set the current averages to the number of repeats
            average_number = 0
            df = pd.DataFrame({type: corr_list})


        # generate new allignment values
        al_list = variation(type)[1]

        # convert the average allignment to a df and the noises as well
        d = {"average_{}".format(average_number): al_list}
        df_new = pd.DataFrame(data = d)

        # print(df.head())
        # print(df_new.head())
        # concate this with old array
        df_w = pd.concat([df, df_new], axis=1, join='inner')


        # write it to csv file
        df_w.to_csv("./averages_{}_{}/N_{}.csv".format(type, model, N), index = False)

    return None

def average_neighbours_variation(n_times, type):
    """
    Creates a file with repeats of how average neighbours vary with noise
    /density in the SVM - the type are the control parameters.
    """

    # list with values of nosie, L and N
    noise_list = list(np.linspace(0, 5, num = 20))
    density_list = list(np.linspace(0.0001, 3, num = 15)) + list(np.linspace(3.5, 10, num = 5))

    if type == "noise":
        corr_list = noise_list
    if type == "density":
        corr_list = density_list

    # for each repeat, add new allignment values to old allignment values
    for repeat in range(n_times):

        # read in data from file if file exists
        try:
            # read in csv as dataframe
            df = pd.read_csv("./neighbours_variation_with_{}_N{}.csv".format(type, N))
            average_number = len(df.columns) - 1

        except IOError as e:
            # set the current averages to the number of repeats
            average_number = 0
            df = pd.DataFrame({type: corr_list})


        # generate new allignment values
        all_neighbours_list = variation(type)[1]

        # convert the average allignment to a df and the noises as well
        d = {"average_{}".format(average_number): all_neighbours_list}
        df_new = pd.DataFrame(data = d)

        # print(df.head())
        # print(df_new.head())
        # concate this with old array
        df_w = pd.concat([df, df_new], axis=1, join='inner')

        # write it to csv file
        df_w.to_csv("./neighbours_variation_with_{}_N{}.csv".format(type, N), index = False)

    return None

def run_to_get_averages(n_times):
    """
    Run this function to get the averages for each of the different setups
    (defined below) for 'n_times' repeats plotted on the same graph.
    """
    global N, L, U
    # initalise the differnt values of L and N that are needed
    # L_list = [3.1, 5, 10, 31.6, 50]
    # N_list = [40, 100, 400, 4000, 10000]
    # U_best = [80, 200, 300, 800, 1500]

    L_list = [3.1, 5]
    N_list = [40, 100]
    U_best = [80, 200]

    # loop over the values in the lists above
    for i in range(len(L_list)):
        # change the values of L and N to the ones from the list
        N = N_list[i]
        L = L_list[i]
        U = U_best[i]

        # check the time of the program
        start = time.time()

        # run the average function n_times
        average_noise_allignment(n_times)

        # time after the function
        end = time.time()
        print("\n-------- time of program: {} -------------\n".format(end - start))

    #show the scatter plot
    #plt.show()

    return None


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

    # particles within r for all i's
    particles_within_r_for_all = []

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

        # for each i particle calculate the number within r
        number_within_r = particles_in_radius(positions[i], positions, velocities)[2]
        particles_within_r_for_all.append(number_within_r)

    np_particles_within_r_for_all = np.array(particles_within_r_for_all)
    average = np.mean(np_particles_within_r_for_all)

    return new_positions, new_vels, average

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
        v_i = velocity[i] + acceleration[i] * delta_t # + noise

        # append to the new_position and velocity list this position/velocity
        new_vel.append(v_i)

    # rescale the magnitude of the speed
    new_vel = rescale(v_mag, new_vel)

    # add the noise
    new_vel = error_force(new_vel)

    return new_vel

def update_acceleration(position_particle, velocity_particle, position_particles, velocity_particles, positions_obj):
    """
    Algorithm which updates the acceleration of each particle
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

    # If using the Vicsek Model get velocity of particles in radius
    if model == "SVM":
        vel_in_r = np.array(particles_in_radius(position_particle, position_particles, velocities_particles)[0])

    # If using kNN neighbours get the velocity of k nearest neighbours
    if model == "kNN":
        vel_in_r = np.array(k_particles(position_particle, position_particles, velocities_particles)[0])

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

def error_force(incoming_velocity):
    """
    Adds a random perturbation to the angle of the incoming velocity and
    returns the new randomly affected acceleration.
    """
    # get the magnitude of the velocity
    incoming_velocity = np.array(incoming_velocity)
    mag = np.sqrt(incoming_velocity.dot(incoming_velocity))

    # change the velocity term to an angle
    acc_angle = np.arctan2(incoming_velocity[1], incoming_velocity[0])

    # add a random perturbation based on 'noise'
    acc_angle += random.uniform(- noise / 2, noise / 2)

    # change back to vector form
    new_vel = angle_to_xy(mag, acc_angle)

    return new_vel

# ----------------------- Reuslts Functions ------------------------------

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

# ----------------------- Visualise Functions ------------------------------

def show_path_2D(start, end, coordinates, coordinates_object, clear = True):
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
    for time_step in range(start, end):
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

def show_allignment_plot(time, allignment):

    # plot time vs allignment for x vs y
    plt.plot(time, allignment, linewidth=1, marker=".", markersize=3)
    plt.xlabel("Time")
    plt.ylabel("Allignment value")
    plt.show()

    return None

def phase_transition(order_parameter_values, control_parameter_values):
    """
    Plots a potential phase diagram between an order parameter, such as alignment
    against a control parameter such as nosie.
    """
    # plot the order parameter on the y axis and the control on the x
    plt.scatter(control_parameter_values, order_parameter_values,
                s = 2, label = "N = {}, L = {}".format(N, L))
    # plt.xlabel("nosie") # these should be changed for other parameters
    # plt.ylabel("allignment") # these should be changed for other parameters
    plt.legend()
    plt.show()

    return None

def plot_average_neighbours(averages_list, U_list):
    x = U_list
    y = averages_list

    plt.scatter(x, y)
    plt.show()

    return 0
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

    # constant for counting how many particles are within radius
    number_within_radius = 0

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

            # also increase number of particles within radius
            number_within_radius += 1

    return velocities_within_r, positions_within_r, number_within_radius

def k_particles(chosen_particle, positions, velocities):
    """
    Checks and records the k closest particles of chosen_particle.
    Returns the velocities and positions of those k particles.
    """


    # array with all indecies of all k particles for positions
    positions_k = []
    velocities_k = []

    # array of new distances considering boundary conditions
    new_distances = []

    # check over all particles in positions
    for index in range(N):

        distance_x, distance_y = per_boun_distance(chosen_particle, positions[index])

        # distance from selected particle to particle with index
        d = np.sqrt(distance_x**2 + distance_y**2)

        # append this distance to array of distances
        new_distances.append(d)

    # Now we need a sorting algorithm (merge)
    for j in range(k+1):
        low = min(new_distances)

        index_k = new_distances.index(low)

        # get the index of the particle for velocity
        velocities_k.append(velocities[index_k])

        # get the index of the particle for position
        # and add position to all positions within r
        positions_k.append(positions[index_k])

        new_distances.pop(index_k)

    return velocities_k, positions_k

def angle_to_xy(magnitude, angle):
    """
    Takes an angle in radians as input as returns x and y poistion for the corresponding angle
    using r as v_mag, a predifined value which is the magnitude of the velocity.
    """
    # get x using x = cos(angle) and y = sin(angle)
    x = magnitude * math.cos(angle)
    y = magnitude * math.sin(angle)

    return [x, y]

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
    d1 = {"a": [i for i in range(10)], "b": [2*i for i in range(10)], "c": [-i for i in range(10)]}
    d2 = {"d": [i for i in range(10)], "e": [2*i for i in range(10)], "f": [-i for i in range(10)]}

    df1 = pd.DataFrame(data = d1)
    df2 = pd.DataFrame(data = d2)

    df3 = pd.concat([df1, df2], axis=1, join='inner')

    print(df1)
    print("\n")
    print(df2)
    print("\n")
    print(df3)

    return None


# run program
start = time.time()
main()
# help()
print("------------------------- Time Taken: {} -------------------".format(time.time() - start))
