"""
Program built to investiagte coding up the physics of object colliding.
"""

import numpy as np
import pandas as pd
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
# CHANGE LATER
mom_inertia = (1/12) * mass_object * 8 # HARD CODE VALUE FOR MOI
noise = 0  # noise added to the acceleration
k = 7 # nearest neighbours

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


# picking a model
model = "SVM" # select SVM for standard Vicsek Model and kNN for nearest neighbours


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
    positions_obj, ang_velocities_obj, accelerations_obj = objects(positions_polygons)

    # append the positions to the positions over time
    pos_part_over_t = [positions]
    vel_part_over_t = [velocities]
    pos_poly_over_t = [positions_polygons]
    ang_vel_obj_over_t = [ang_velocities_obj]

    # get the allignment
    align_start = allignment(velocities)

    # make a list which will contain the sum of distance fromt the centre
    SD_list = []

    # update the position for 10 times
    for i in range(U):

        # call update to get the new positions of the particles
        positions, velocities = update_system(positions, velocities, positions_obj, positions_polygons)

        # update the positions of the objects
        positions_polygons, ang_velocities_obj = update_system_object(positions_polygons, positions_obj, ang_velocities_obj,
                                                                      positions, velocities)

        # append in positions over time
        pos_part_over_t.append(positions)
        vel_part_over_t.append(velocities)
        pos_poly_over_t.append(positions_polygons)
        ang_vel_obj_over_t.append(ang_velocities_obj)

        # get the SD for this loop
        SD = SD_COM(positions)
        SD_list.append(SD)

    align_end = allignment(velocities)

    # plot the movment of the particles if plot is set to true
    if plot == True:
        show_path_2D(U - U, U, pos_part_over_t, pos_poly_over_t, clear = True)

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
        force_particles += chate_rep_att_force(position_particle, particle)

    # calcualte force due to the objects
    for object in range(len(positions_obj)):
        force_object += contact_force_particle(polygons[object], positions_obj[object], velocity_particle, position_particle)


    new_acceleration = (alpha * force_particles + beta * force_object +
    gamma * allignment_force(position_particle, velocity_particle, position_particles, velocity_particles)) / mass_par

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
        ang_acceleration = update_ang_acceleration_object(polygons[i], positions_obj[i], positions_obj, position_particles, velocity_particles)
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

def update_ang_acceleration_object(polygon, position_obj, positions_obj, position_particles, velocity_particles):
    """
    Algorithm which updates the acceleration of the com of the object
    """
    # define two inital forces dependent on the particles and on hte object
    torque_particles = 0

    # loop through each particle and calculate the repulsive force from the particle
    for particle in range(len(position_particles)):
        torque_particles +=  torque_force(polygon, position_obj, velocity_particles[particle],position_particles[particle])

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

def torque_force(polygon, position_obj, velocity_particle, position_particle):
    """
    Calcualte the torque on an object due to a particle hitting it.
    """
    # make the lists np arrays
    position_obj = np.array(position_obj)
    position_particle = np.array(position_particle)

    # get r from centroid and force
    r = position_particle - position_obj
    force = contact_force_object(polygon, position_obj, velocity_particle, position_particle)

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
    torque = torque[2]


    # # get the respective magnitudes
    # r_mag = np.sqrt(r.dot(r))
    # force_mag = np.sqrt(force.dot(force))
    #
    # # compute t = r*(F*sin(theta)) note: teh force already takes into account for the angle
    # torque = r_mag * force_mag * math.sin(angle)

    # account for torque in other direction
    # check if force and radius are in oppossite direction: DO THIS WITH THE ANGLE
    # direction = r
    # if sdfdf:
    #     torque = torque * -1

    return torque

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


    return -Force

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

    poly = polygon()
    pol_ct = centroid(poly)

    vel_part = [1, -1]
    pos_part = [L/2 - 1 - 0.001, L/2 - 0.5]

    poly = np.array(poly)
    # get the points of the polygon to plot it
    x, y = poly.T

    # print(x, y)

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # print(x, y)

    # plot the polygon

    new_pos = update_position_object_vertex(poly, pol_ct, math.pi/4)
    new_pos = np.array(new_pos)

    print(poly)
    print(new_pos)

    # get the points of the polygon to plot it
    x_2, y_2 = new_pos.T

    x_2 = np.append(x_2, x_2[0])
    y_2 = np.append(y_2, y_2[0])


    # plot the polygon
    plt.plot(x , y, c = 'b')
    plt.plot(x_2 , y_2, c = 'r')
    # plt.scatter(pos_part[0], pos_part[1], s = 3, c =  'r')
    plt.show()
    return None


# run program
start = time.time()
main()
# help()
print("------------------------- Time Taken: {} -------------------".format(time.time() - start))
