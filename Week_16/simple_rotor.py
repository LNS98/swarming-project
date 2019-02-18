"""
Program built to investiagte coding up the physics of object colliding.
"""
from __future__ import division

import numpy as np
from scipy.stats import moment
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.cm as cm
from shapely.geometry import LineString, Point, LinearRing, Polygon
import time
from multiprocessing import Pool

# constants used in the program
bound_cond = True   # set the boundry conditions on or off
<<<<<<< HEAD
L = 3  # size of the box
N = 70  # number of particles
=======
L = 5 # size of the box
N = 50  # number of particles
>>>>>>> 705323b0bd75dfaba32b90f7c410c8c2793663b2
k = 2 # nearest neighbours

M = 1   # number of objects
v_mag = 0.05      # total magnitude of each particle velocity
delta_t = 1     # time increment
mass_par = 1 # masss of the particles
<<<<<<< HEAD
b = 1 # outer size of radius of object
=======
b = L*0.2 # outer size of radius of object
>>>>>>> 705323b0bd75dfaba32b90f7c410c8c2793663b2

mass_object = 1000 # masss of the object
mom_inertia = (1/3) * mass_object # PERHAPS CHANGE A BIT BUT ITS JUST A DAMPING TERM SO DON'T WORRY TOO MUCH

# distance metrics in the code
r = 1.0   # radius of allignment
r_c = 0.05 # radius within repulsion
r_e = 0.5 # radius of equilibrium between the particles
r_a = 0.8 # radius when attraction starts
r_o = v_mag # radius of attraction between the particels and the objects

# force parrameters
alpha = 1 # stregnth of repulsive force between to the particles
beta = 1 # stregnth of the force due to the objects on the particles
gamma = 0 # stregnth of allignment
fric_force = 0.2  # frictional forrce of the object when rotating
noise = 0  # noise added to the velocity


# picking a model
model = "kNN" # select SVM for standard Vicsek Model and kNN for nearest neighbours


U = 1000   # number of updates
dimensions = 2   # dimensions
time_pause = 1e-7 # time pause for interactive graph



def main():

    # make 1 complete run of the system
    poolie = Pool(processes = 4)

    data = poolie.map(one_run, [True for i in range(1)])

    # wait till the processes are finished
    poolie.close()
    poolie.join()


    data_np = np.array(data)
    ave = np.mean(data_np)
    s_d = np.std(data_np)

    print("average is: {} with SD of {}".format(ave, s_d))

    return 0

# ----------------------- Whole system Functions ---------------------------------

def one_run(plot = False):
    """
    One simulation of a total run by the system.
    """

    # produce the polygons (verticeies of the polygon)
    positions_polygons = [polygon([L/2, L/2], L*0.2, math.pi / 3, 15) for i in range(M)]

    # fill up a box with particles and objects
    positions, velocities, accelerations = pop_box(positions_polygons)
    # MAYBE MAKE THIS POP_OBJECTS
    # returns positions, velocities, accelerations of com of objects
    positions_obj, ang_velocities_obj, accelerations_obj = objects(positions_polygons)

    # append the positions to the positions over time
    angle_over_t = [0]
    pos_part_over_t = [positions]
    vel_part_over_t = [velocities]
    pos_poly_over_t = [positions_polygons]
    ang_vel_obj_over_t = [ang_velocities_obj]

    # get the allignment
    align_start = allignment(velocities)


    # update the position for 10 times
    for i in range(U):

        # print("timestep: {}".format(i))
        # call update to get the new positions of the particles
        positions, velocities = update_system(positions, velocities, positions_obj, positions_polygons)

        # update the positions of the objects
        positions_polygons, ang_velocities_obj = update_system_object(positions_polygons, positions_obj, ang_velocities_obj,
                                                                      positions, velocities)

        # get the angle variaition due to the ang velocity
        new_angle = angle_over_t[-1]  + ang_velocities_obj[0] * delta_t

        # append in positions over time
        pos_part_over_t.append(positions)
        vel_part_over_t.append(velocities)
        pos_poly_over_t.append(positions_polygons)
        ang_vel_obj_over_t.append(ang_velocities_obj)
        angle_over_t.append(new_angle)


    ang_velocities_obj_end = ang_velocities_obj
    align_end = allignment(velocities)

    # plot the movment of the particles if plot is set to true
    if plot == True:
        show_path_2D(0, U, pos_part_over_t, pos_poly_over_t, clear = True)

    return angle_over_t[-1]

# ----------------------- Building the objects Functions ---------------------------------

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

# ----------------------- System Functions ---------------------------------

def pop_box(polygons):
    """
    Function which creates one particle with a ramdom position and velocity
    in a square of dimensions L by L.
    """

    # will hold each posi/vel/acc for each particle in the system
    positions = []
    velocities = []
    accelerations = []

    count = 0
    while count < N:
        # lsit containing positions and velocities at random
        # position can't be in the box
        init_position = [random.uniform(0, L) for i in range(dimensions)] # IMPROVE

        #  check if point is within the polygon any of the polygons
        inside = False
        for poly in polygons:
            cnt = centroid(poly)
            cond = (cnt[0] - init_position[0]) ** 2 + (cnt[1] - init_position[1]) ** 2
            if cond < (L*0.2 + 5 * v_mag * delta_t) ** 2:
                inside = True
                break
        if inside == True:
            continue

        init_velocity = [random.uniform(-1, 1) for i in range(dimensions)]
        init_acceleration = [0 for i in range(dimensions)]

        # append the positions to the bigger lists
        positions.append(init_position)
        velocities.append(rescale(v_mag, init_velocity))
        accelerations.append(init_acceleration)

        # add 1 to the count
        count += 1

    return positions, velocities, accelerations

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
        force_particles += part_repulsive_force(position_particle, particle)

    # calcualte force due to the objects
    for object in range(len(positions_obj)):
        force_object -= contact_force(polygons[object], positions_obj[object], position_particle, velocity_particle)


    new_acceleration = (alpha * force_particles + beta * force_object +
    gamma * allignment_force(position_particle, velocity_particle, position_particles, velocity_particles)) / mass_par
    # print(new_acceleration)
    return new_acceleration

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

    new_acceleration = torque_particles / mom_inertia

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

# ----------------------- Forces Functions ------------------------------


def torque_force(polygon, position_obj, ang_vel_object, velocity_particle, position_particle):
    """
    Calcualte the torque on an object due to a particle hitting it.
    """
    force = contact_force(polygon, position_obj, position_particle, velocity_particle)

    # make the lists np arrays
    position_obj = np.array(position_obj)
    position_particle = np.array(position_particle)

    # get r from centroid and force
    r = position_particle - position_obj

    # get the angle between r and force
    v1_u = rescale(1, r)
    v2_u = rescale(1, force)
    angle =  np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    # insert 0s in the third dimension of the torque
    r = np.insert(r, 2, 0)
    force = np.insert(force, 2, 0)

    # get the torque from t = r X F
    torque = np.cross(r, force)

    # get only th emagnitude of the force
    torque = torque[2] - fric_force * ang_vel_object

    return torque

def obj_repulsive_force(particle_position, polygon):
    """
    calculates the force used in the repulsive_force function. As per chate 2008
    """
    # make the polygon a linear ring
    poly = LinearRing(polygon)
    # create a particle moving straight down
    point = Point(particle_position)

    # get the closest point on polygon to particle
    d = poly.project(point)
    p = poly.interpolate(d)
    closest_point = list(p.coords)[0]

    # call that j and call particle_position i
    i = particle_position
    j = closest_point

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
       magnitude = 1 /(1 + math.exp(distance/ r_o))

    except OverflowError as err:
        magnitude = 0

    # get the x direction of the force
    F_x = (magnitude * distance_x) / distance

    # get the y direction of the force
    F_y = (magnitude * distance_y) / distance

    return np.array([F_x, F_y])

def part_repulsive_force(i, j):
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
        magnitude = 1e6

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

def contact_force(polygon, position_obj, position_particle, velocity_particle):
    """
    Contact force between object and particle.
    """

    # make the polygon a linear ring and a polygon
    poly = LinearRing(polygon)

    # create a particle moving straight down
    point = Point(position_particle)

    # get the distance between the object and the particle
    dist = point.distance(poly)

    # check if the particle is not touching
    if (dist > 5 * v_mag * delta_t):
        return np.array([0, 0])

    # get the closest point
    d = poly.project(point)
    p = poly.interpolate(d)
    closest_point = list(p.coords)[0]

    # now you have the points,get the vecetor normal to the plane
    n = rescale(1, [position_particle[0] - closest_point[0], position_particle[1] - closest_point[1]])
    # get the value of n, the normalised normal vector to the surface of reflection
    n = np.array(n)

    if (dist < 3*v_mag*delta_t):
        return obj_repulsive_force(position_particle, polygon)

    # define the magntiude of the vector force
    magnitude = np.dot(velocity_particle, n)

    # get the force in the direction of the surface normal
    Force = magnitude * n

    return Force

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
    com = np.array(centroid(position_particles))

    sum = 0
    # loop over each particle in the positions
    for particle in position_particles:
        sum += distance_fun(particle, com)

    return sum

# ----------------------- Visualise Functions ------------------------------

def show_path_2D(start, end, coordinates, polygons, clear = True):
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
            plt.scatter(coordinates[time_step][i][0], coordinates[time_step][i][1], s = 1, color = 'r')

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
                # plt.scatter(polygons_com[time_step][i][0], polygons_com[time_step][i][1], s = 5, color = 'g')

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
    plt.clf()
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

def clockwiseangle(point):
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

def area(pts):
    'Area of cross-section.'

    if pts[0] != pts[-1]:
      pts = pts + pts[:1]

    x = [ c[0] for c in pts ]
    y = [ c[1] for c in pts ]
    s = 0

    for i in range(len(pts) - 1):
        s += x[i]*y[i+1] - x[i+1]*y[i]

    return s/2

def centroid(pts):
        'Location of centroid.'

        # check if the last point is the same as the first, if nots so 'close' the polygon
        if pts[0] != pts[-1]:
            pts = pts + pts[:1]

        # get the x and y points
        x = [c[0] for c in pts]
        y = [c[1] for c in pts]

        # initialise the x and y centroid to 0 and get the area of the polygon
        sx = sy = 0
        a = area(pts)

        for i in range(len(pts) - 1):
            sx += (x[i] + x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
            sy += (y[i] + y[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])

        return [sx/(6*a), sy/(6*a)]

def inertia(pts):
    'Moments and product of inertia about centroid.'

    if pts[0] != pts[-1]:
      pts = pts + pts[:1]

    x = [c[0] for c in pts]
    y = [c[1] for c in pts]

    sxx = syy = sxy = 0
    a = area(pts)
    cx, cy = centroid(pts)

    for i in range(len(pts) - 1):
      sxx += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
      syy += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
      sxy += (x[i]*y[i+1] + 2*x[i]*y[i] + 2*x[i+1]*y[i+1] + x[i+1]*y[i])*(x[i]*y[i+1] - x[i+1]*y[i])

    return [sxx/12 - a*cy**2, syy/12 - a*cx**2]

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

    poly = polygon([L/2, L/2], 3, -math.pi / 4, 8)
    pos_obj = centroid(poly)
    pos_part = [L/ 2 - 2 + 0.8, L/2 + 0.4]
    vel_part = [1, 0]

    force = contact_force_particle(poly, pos_obj, pos_part, vel_part)

    print(force)

    poly = np.array(poly)

    x, y = poly.T

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    plt.plot(x, y)
    plt.scatter(x, y, c = 'r')
    plt.scatter(pos_part[0], pos_part[1], c = 'b', s = 2)
    plt.show()
    return None

# if __name__ == "__main__":
#     start = time.time()
#     main()
#     # help()
#     print("------------------------- Time Taken: {} -------------------".format(time.time() - start))
