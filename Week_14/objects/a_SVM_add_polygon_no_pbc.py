"""
Program built to investiagte coding up the physics of object colliding.
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.cm as cm
from shapely.geometry import LineString, Point, LinearRing, Polygon
import time

# constants used in the program
bound_cond = True   # set the boundry conditions on or off
L = 10  # size of the box
N = 30  # number of particles
M = 1   # number of objects
v_mag = 0.05      # total magnitude of each particle velocity
delta_t = 1     # time increment
mass_par = 1 # masss of the particles
mass_object = 10 # masss of the object
noise = 0  # noise added to the acceleration

# distance metrics in the code
r = 1.0   # radius of allignment
r_c = 0.2 # radius within repulsion
r_e = 0.5 # radius of equilibrium between the particles
r_a = 0.8 # radius when attraction starts
r_o = 0.05 # radius of attraction between the particels and the objects

# force parrameters
alpha = 0 # stregnth of repulsive force due to the particles
beta = 1 # stregnth of the force due to the objects
gamma = 0 # stregnth of allignment

U = 500   # number of updates
dimensions = 2   # dimensions
time_pause = 1e-10 # time pause for interactive graph




def main():

    # make 1 complete run of the system
    ali_end, SD_list = one_run(plot = True)
    print("alignment: {}".format(ali_end))

    SD_graph(SD_list)

    return 0

# ----------------------- Whole system Functions ---------------------------------

def one_run(plot = False):
    """
    One simulation of a total run by the system.
    """

    # produce the polygons (verticeies of the polygon)
    positions_polygons = [polygon() for i in range(M)]

    # fill up a box with particles and objects
    positions, velocities, accelerations = pop_box()
    # MAYBE MAKE THIS POP_OBJECTS
    # returns positions, velocities, accelerations of com of objects
    positions_obj, velocities_obj, accelerations_obj = objects(positions_polygons)

    # append the positions to the positions over time
    pos_part_over_t = [positions]
    vel_part_over_t = [velocities]
    pos_obj_over_t = [positions_obj]
    pos_poly_over_t = [positions_polygons]
    vel_obj_over_t = [velocities_obj]

    # get the allignment
    align_start = allignment(velocities)

    # make a list which will contain the sum of distance fromt the centre
    SD_list = []

    # update the position for 10 times
    for i in range(U):

        # call update to get the new positions of the particles
        positions, velocities = update_system(positions, velocities, accelerations, positions_obj, positions_polygons)

        # update the positions of the objects
        positions_polygons, positions_obj, velocities_obj = update_system_object(positions_polygons, positions_obj, velocities_obj, accelerations_obj,
                                                                     positions, velocities)

        # append in positions over time
        pos_part_over_t.append(positions)
        vel_part_over_t.append(velocities)
        pos_obj_over_t.append(positions_obj)
        pos_poly_over_t.append(positions_polygons)
        vel_obj_over_t.append(velocities_obj)

        # get the SD for this loop
        SD = SD_COM(positions)
        SD_list.append(SD)

    align_end = allignment(velocities)

    # plot the movment of the particles if plot is set to true
    if plot == True:
        show_path_2D(U - U, U, pos_part_over_t, pos_poly_over_t, pos_obj_over_t, clear = True)

    return align_end, SD_list

# ----------------------- Building the objects Functions ---------------------------------

def polygon():
    """
    Define the polygon from the points on the verticies.
    """
    # regular polygon for testing
    # lenpoly = 5
    # polygon = np.array([[random.random() + L/2, random.random() + L/2] for x in range(4)])

    polygon =[[L/2 - 1, L/2 - 1], [L/2 + 1, L/2 - 1], [L/2 + 1, L/2 + 1], [L/2 - 1, L/2 + 1]]

    return polygon

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
        init_position = [random.choice([random.uniform(0, L/2 - 1), random.uniform(L/2 + 1, L)]) for i in range(dimensions)] # IMPROVE
        init_velocity = [random.uniform(-1, 1) for i in range(dimensions)]
        init_acceleration = [0 for i in range(dimensions)]

        # append the positions to the bigger lists
        positions.append(init_position)
        velocities.append(rescale(v_mag, init_velocity))
        accelerations.append(init_acceleration)

    return positions, velocities, accelerations

def objects(polygons):
    """
    Create a set of M objects, defining them just by there centre of mass.
    As of now they are basically particles of different species.
    """

    # will hold each posi/vel/acc for each particle in the system
    positions = []
    velocities = []
    accelerations = []

    for i in range(M):
        # lsit containing positions and velocities at random
        init_position = get_com(polygons[i]).tolist()
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

def update_system(positions, velocities, accelerations, positions_obj, polygons):
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
    new_vel = error_force(new_vel)

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
        force_particles += chate_rep_att_force(position_particle, particle)

    # calcualte force due to the objects
    for object in range(len(positions_obj)):
        force_object += contact_force_particle(polygons[object], positions_obj[object], velocity_particle, position_particle)


    new_acceleration = (alpha * force_particles + beta * force_object +
    gamma * allignment_force(position_particle, velocity_particle, position_particles, velocity_particles)) / mass_par

    return new_acceleration

# ----------------------- Update Functions for objects ------------------------------

def update_system_object(polygons, positions_obj, velocities_obj, accelerations_obj, position_particles, velocity_particles):
    """
    Updates the positons and velocities of ALL the objects in a system.
    """
    # lists which will contain the updated values
    new_positions = []
    new_vels = []
    new_polygons = []

    # loop through each index in the positions, vel, acc
    for i in range(len(positions_obj)):
        # update centre of Mass

        # get the acceleration based on the positions of the particles
        acceleration = update_acceleration_object(polygons[i], positions_obj[i], positions_obj, position_particles, velocity_particles)
        # call update to get the new value
        new_vel = update_velocity_object(velocities_obj[i], acceleration)
        new_pos = update_position_object(positions_obj[i], new_vel)
        # append it to the new values
        new_positions.append(new_pos)
        new_vels.append(new_vel)

        # update the position of the verticies
        new_vers = update_position_object_vertex(polygons[i], new_vel)
        new_polygons.append(new_vers)


    return new_polygons, new_positions, new_vels

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

        # chek for boundry conditions
        if bound_cond == True:
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

def update_acceleration_object(polygon, position_obj, positions_obj, position_particles, velocity_particles):
    """
    Algorithm which updates the acceleration of the com of the object
    """
    # define two inital forces dependent on the particles and on hte object
    force_object = np.array([0., 0.])
    force_particles = np.array([0., 0.])

    # loop through each particle and calculate the repulsive force from the particle
    for particle in range(len(position_particles)):
        force_particles +=  contact_force_object(polygon, position_obj, velocity_particles[particle],position_particles[particle])

    # calcualte force due to the objects
    for object in positions_obj:
        if object == position_obj:
            continue
        force_object += obj_repulsive_force(position_obj, object)

    new_acceleration = (beta * force_particles + alpha * force_object) / mass_object

    return new_acceleration

def update_position_object_vertex(polygon, velocities_obj):
    """
    Update the location of a particle and returns the new location.
    """

    # get the points of the polygon to plot it
    polygon = np.array(polygon)
    x, y = polygon.T

    # add the velocity in that dimension to the position (times delta_t)
    new_x = x + velocities_obj[0] * delta_t
    new_y = y + velocities_obj[1] * delta_t

    # reconvert to the polygon form
    new_pos = list(np.array([list(new_x), list(new_y)]).T)

    return new_pos


# ----------------------- Forces Functions ------------------------------

def contact_force_object(polygon, position_obj, velocity_particle, position_particle):
    """
    Contact force between object and particle.
    """

    # make the polygon a linear ring
    poly = LinearRing(polygon)

    # create a particle moving straight down
    point = Point(position_particle)

    # inside = point.within(poly_2)
    # get the distance between the object and the particle
    dist = point.distance(poly)

    if dist > 0.05:
        return np.array([0, 0])

    d = poly.project(point)
    p = poly.interpolate(d)
    closest_point = list(p.coords)[0]

    # now you have the points, try to get the vecetor normal to the plane
    n = rescale(1, [position_particle[0] - closest_point[0], position_particle[1] - closest_point[1]])

    # get the value of n, the normalised normal vector to the surface of reflection
    n = np.array(n)
    v_1 = np.array(velocity_particle)

    # calcualte the wanted velocity for the velocity after reflection
    v_2 = -(2 * (np.dot(v_1, n)) * n - v_1)

    Force = v_2 - v_1


    return - Force

def contact_force_particle(polygon, position_obj, velocity_particle, position_particle):
    """
    Contact force between object and particle.
    """

    # make the polygon a linear ring
    poly = LinearRing(polygon)
    # create a particle moving straight down
    point = Point(position_particle)

    # inside = point.within(poly_2)
    # get the distance between the object and the particle
    dist = point.distance(poly)

    if dist > 0.05:
        return np.array([0, 0])

    d = poly.project(point)
    p = poly.interpolate(d)
    closest_point = list(p.coords)[0]

    # now you have the points, try to get the vecetor normal to the plane
    n = rescale(1, [position_particle[0] - closest_point[0], position_particle[1] - closest_point[1]])

    # get the value of n, the normalised normal vector to the surface of reflection
    n = np.array(n)
    v_1 = np.array(velocity_particle)

    # calcualte the wanted velocity for the velocity after reflection
    v_2 = -(2 * (np.dot(v_1, n)) * n - v_1)

    Force = v_2 - v_1


    return Force

def obj_repulsive_force(i, j):
    """
    calculates the force used in the repulsive_force function. As per chate 2008
    """
    if bound_cond == True:
        # calculate the distance between the points
        distance_x, distance_y = per_boun_distance(i, j)
        # calcualte the magnitude of the distance between the points
        distance = (distance_x ** 2 + distance_y ** 2) ** (1/2)

    else:
        distance_x, distance_y = j[0] - i[0], j[1] - i[1]
        distance = distance_fun(i, j)

    try:
        # magnitude of force
        magnitude = -1 /(1 + math.exp(distance/ r_o))

    except OverflowError as err:
        magnitude = 0

    # get the x direction of the force
    F_x = (magnitude * distance_x) / distance

    # get the y direction of the force
    F_y = (magnitude * distance_y) / distance

    return np.array([F_x, F_y])

def inverse_force(i, j):
    """
    (1/r)^2 repulsive force
    """
    if bound_cond == True:
        # calculate the distance between the points
        distance_x, distance_y = per_boun_distance(i, j)
        # calcualte the magnitude of the distance between the points
        distance = (distance_x ** 2 + distance_y ** 2) ** (1/2)

    else:
        distance_x, distance_y = j[0] - i[0], j[1] - i[1]
        distance = distance_fun(i, j)

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
    # check for bounfy conditions
    if bound_cond == True:
        # calculate the distance between the points
        distance_x, distance_y = per_boun_distance(i, j)
        # calcualte the magnitude of the distance between the points
        distance = (distance_x ** 2 + distance_y ** 2) ** (1/2)

    else:
        distance_x, distance_y = j[0] - i[0], j[1] - i[1]
        distance = distance_fun(i, j)

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

def SD_COM(position_particles):
    """
    Calcualte the sum of scalar distance of all the particles from the centre of mass of
    the particles.
    """

    # calculate the centre of mass of the object
    com = get_com(position_particles)

    sum = 0
    # loop over each particle in the positions
    for particle in position_particles:
        sum += distance_fun(particle, com)

    return sum

# ----------------------- Visualise Functions ------------------------------

def show_path_2D(start, end, coordinates, polygons, polygons_com, clear = True):
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
                polygon = np.array(polygons[time_step][i])
                # get the points of the polygon to plot it
                x, y = polygon.T

                # print(x, y)

                x = np.append(x, x[0])
                y = np.append(y, y[0])

                # print(x, y)

                # plot the polygon
                plt.plot(x , y)
                plt.scatter(polygons_com[time_step][i][0], polygons_com[time_step][i][1], s = 5, color = 'g')

            if bound_cond == True:
                plt.axis([0, L, 0, L])
            plt.axis([0, L, 0, L])
            # plt.axis([-L*2, L*2, -L*2, L*2])

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
    plt.xlabel("nosie") # these should be changed for other parameters
    plt.ylabel("allignment") # these should be changed for other parameters
    plt.legend()
    plt.show()

    return None

def SD_graph(SD_list):
    """
    Graph the results of the sum of distances.
    """
    # get the x values, the timesteps
    x = [i for i in range(U)]

    # plot the results
    plt.scatter(x, SD_list, s = 3)
    plt.xlabel("Time Step")
    plt.ylabel("Sum of Distance from Centre of Mass")
    plt.show()

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

            if bound_cond == True:
                inside_distance = abs(position_particle[i] - position_particles[index][i])
                wrap_distance = L-inside_distance
                distance = min(inside_distance, wrap_distance)
            else:
                distance = abs(position_particle[i] - position_particles[index][i])

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

def get_com(particle_positions):
    """
    Get the centre of mass of the particles given
    """
    # the centre of mass is just the average in each dimension

    # array containing which will contain the com
    com = []
    number_of_particles = len(particle_positions)

    # loop over each dimension
    for i in range(dimensions):
        # sum variable for the given dimensions
        sum_i = 0

        # loop over each particle
        for particle in range(number_of_particles):
            sum_i += particle_positions[particle][i]

        # now average the sum over N and append to the com
        sum_i = sum_i / number_of_particles
        com.append(sum_i)

    return np.array(com)

def distance_fun(pos1, pos2):
    """
    Calculate the distance between the points
    """
    # get the two arrays as np arrays, easier to do calculations
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)

    # get the distance
    distance = pos2 - pos1

    # distance is the same as the magnitude
    dist = np.sqrt(distance.dot(distance))

    return dist

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

# ----------------------- Test functins  Functions ------------------------------


def help():
    """
    Funciton used for different reasons.
    """

    # get a polygon
    poly_list = polygon()

    # make the polygon a linear ring
    poly_1 = Polygon(poly_list)
    poly_2 = LinearRing(poly_list)

    # create a particle moving straight down
    part_pos = [5, 5]
    point = Point(part_pos)


    distance = poly_2.distance(point)
    inside = point.within(poly_1)

    boundary = poly_1.boundary()

    print(boundary)


    # # contact_force_particle(polygon, position_obj, velocity_particle, position_particle)
    #
    # # plot the polygon
    # x_poly, y_poly = poly.coords.xy
    # plt.plot(x_poly, y_poly)
    # plt.scatter(part_pos[0], part_pos[1])
    # plt.show()

    return None


# run program
start = time.time()
main()
# help()
print("------------------------- Time Taken: {} -------------------".format(time.time() - start))
