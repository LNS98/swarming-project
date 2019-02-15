
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



L = 30  # size of the box
U = 100   # timesteps



def pop_box(polygons):
    """
    Function which populates the box by taking in a rotor and making
    sure that particles are not within it
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
            if cond < (poly.outer_r + 5 * v_mag * delta_t) ** 2:
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




#------------------ Particle Update Functions ---------------------
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
        force_object -= contact_force_particle(polygons[object], positions_obj[object], position_particle, velocity_particle)


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

# ------------------------Forces -----------------------------------------------
def torque_force(polygon, position_obj, ang_vel_object, velocity_particle, position_particle):
    """
    Calcualte the torque on an object due to a particle hitting it.
    """
    # make the lists np arrays
    position_obj = np.array(position_obj)
    position_particle = np.array(position_particle)

    # get r from centroid and force
    r = position_particle - position_obj
    force = contact_force_object(polygon, position_obj, position_particle, velocity_particle)

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
    torque = torque[2] - 1 * ang_vel_object

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

def contact_force_particle(polygon, position_obj, position_particle, velocity_particle):
        """
        Contact force between object and particle.
        """

        # make the polygon a linear ring and a polygon
        poly = LinearRing(polygon)
        poly_poly = Polygon(polygon)

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

        # make sure normal vector is always outward pointing
        if poly_poly.contains(p):
            n = -1 * n

        if dist < 4.5 * v_mag * delta_t:
            # basically inifinite force
            magnitude = 1e6

        else:
            # define the magntiude of the vector force
            magnitude = abs(np.dot(velocity_particle, n))

        # check if the point is inside and if so revert the firection of the normal as this should always be outside

        # get the force in the direction of the surface normal
        Force = magnitude * n


        return Force

def contact_force_object(polygon, position_obj, position_particle, velocity_particle):
    """
    Contact force between object and particle.
    """

    # make the polygon a linear ring and a polygon
    poly = LinearRing(polygon)
    poly_poly = Polygon(polygon)

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

    # define the magntiude of the vector force
    magnitude = np.dot(velocity_particle, n)

    # check if the point is inside and if so revert the firection of the normal as this should always be outside
    if poly_poly.contains(p):
        n = -1 * n

    # get the force in the direction of the surface normal
    Force = magnitude * n

    return Force

#---------------------- Functions which help main functions ----------
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

def centroid(pts):
        'Location of centroid of the rotor'

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
