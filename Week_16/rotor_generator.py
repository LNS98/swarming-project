"""
This file has the purpose of generating a random rotor
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random

from shapely.geometry import LinearRing

from simple_rotor import *


class Rotor:

    # this will not change for all rotors
    outer_r = b
    origin = [L/2, L/2]

    # Instances are specific to each object
    def __init__(self, inner_r, spikes, angle):
        self.inner_r = inner_r
        self.spikes = spikes
        self.angle = angle
        self.fit = 0

    def description(self):
        print( "inner_r: {}, outer_r: {}, spikes: {}, angle: {}, fitness: {}".format(
        self.inner_r, self.outer_r, self.spikes, self.angle, self.fit))
        return None

    def vertices(self):
        """
        This will take in all of the set values of the rotor
        including, the inner_r, the spikes, the angle shift,
        the outer_r and return the vertices of each so that
        it can be plotted
        """
        refvec = (0, 1)

        def clockwiseangle_and_distance(point):
            """
            Finds angle between point and ref vector ffor sortinf points in rotor
            in order of angles
            """


            # Vector between point and the origin: v = p - o
            vector = [point[0]-self.origin[0], point[1]-self.origin[1]]

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

        x_0 = self.origin[0]  # centre of circle
        y_0 = self.origin[1]  # centre of circle

        for i in range(self.spikes):
            value_out = [x_0 + self.outer_r * np.cos(2 * np.pi * i / self.spikes), y_0 + self.outer_r * np.sin(2 * np.pi * i / self.spikes)]
            value_in = [x_0 + self.inner_r * np.cos(self.angle + 2 * np.pi * i / self.spikes), y_0 + self.inner_r * np.sin(self.angle + 2 * np.pi * i / self.spikes)]
            pts_out.append(value_out)
            pts_in.append(value_in)

        sorted_out = sorted(pts_out, key=clockwiseangle_and_distance)
        sorted_in = sorted(pts_in, key=clockwiseangle_and_distance)

        for i in range(len(sorted_in)):
            pts_tot.append(sorted_out[i])
            pts_tot.append(sorted_in[i])

        return pts_tot

    def get_x_y(self):
        points_to_plot = self.vertices()

        points_to_plot = np.array(points_to_plot)
        x, y = points_to_plot.T

        x = np.append(x, x[0]).tolist()
        y = np.append(y, y[0]).tolist()

        return x, y

    def validation(self):
        """
        Returns False for crossing shapes
        """
        points = self.vertices()
        shape = LinearRing(points)

        return shape.is_valid

    def fitness(self):
        """
        returns the final angle which the rotor has moved for a
        set amount of time T_final
        """
        # make 1 complete run of the system
        poolie = Pool(processes = 24)

        data = poolie.map(one_run, [i for i in range(50)])

        # wait till the processes are finished
        poolie.close()
        poolie.join()

        data_np = np.array(data)
        ave = np.mean(data_np)

        self.fit = ave

        return None

    def mutate(self, property):
        """
        Function which changes one of the instances of the object
        """

        if property == self.inner_r:
            property =  random.uniform(0.5, 8.5)

        if property == self.spikes:
            property = random.randint(4,15)

        if property == self.angle:
            property = random.uniform(0, 2 * math.pi)

        return None

def random_rotor():
    """
    Function which generates a rotor with random values for all variables
    Given the outer radius
    Given the center of shape
    It returns the rotor object
    """

    inner_r = random.uniform(0.1, L*0.2)
    spikes = random.randint(4,15)
    angle = random.uniform(0, 2 * math.pi)

    rotor = Rotor(inner_r, spikes, angle)

    # If it is a bad rotor, return false
    if not rotor.validation():
        return random_rotor()

    return rotor


def one_run(rotor, i):


    #get vertices of rotor
    vertices = [rotor.vertices() for i in range(1)]

    # fill up a box with particles and objects
    positions, velocities, accelerations = pop_box(vertices)

    # returns positions, velocities, accelerations of com of objects
    positions_obj, ang_velocities_obj, accelerations_obj = objects(vertices)

    # append the positions to the positions over time
    angle_over_t = [0]
    pos_part_over_t = [positions]
    vel_part_over_t = [velocities]
    pos_poly_over_t = [vertices]
    ang_vel_obj_over_t = [ang_velocities_obj]

    # get the allignment
    align_start = allignment(velocities)

    # update the position for 10 times
    for i in range(U):

        # call update to get the new positions of the particles
        positions, velocities = update_system(positions, velocities, positions_obj, vertices)

        # update the positions of the objects
        vertices, ang_velocities_obj = update_system_object(vertices, positions_obj, ang_velocities_obj,
        positions, velocities)

        # get the angle variaition due to the ang velocity
        new_angle = angle_over_t[-1]  + ang_velocities_obj[0] * delta_t

        # append in positions over time
        pos_part_over_t.append(positions)
        vel_part_over_t.append(velocities)
        pos_poly_over_t.append(vertices)
        ang_vel_obj_over_t.append(ang_velocities_obj)
        angle_over_t.append(new_angle)


    ang_velocities_obj_end = ang_velocities_obj
    align_end = allignment(velocities)

    return angle_over_t[-1]


#
# rotor = random_rotor()
# describe= rotor.description()
# x, y = rotor.get_x_y()
# plt.plot(x, y)
# plt.show()
