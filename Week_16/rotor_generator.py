"""
This file has the purpose of generating a random rotor
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import LinearRing

L = 10

class Rotor:

    # this will not change for all rotors
    outer_r = 9
    origin = [L/2, L/2]

    # Instances are specific to each object
    def __init__(self, inner_r, spikes, angle):
        self.inner_r = inner_r
        self.spikes = spikes
        self.angle = angle

    def description(self):
        return "rotor has {} as inner rad, {} as outer rad, {} spikes and angle {}.".format(
        self.inner_r, self.outer_r, self.spikes, self.angle)


    # This method returns the correct vertices of the rotor in order
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
            pts_tot.append(sorted_in[i])
            pts_tot.append(sorted_out[i])

        return pts_tot

    # method which returns two arrays [x] and [y] for plotting the rotor
    def get_x_y(self):
        points_to_plot = self.vertices()


        points_to_plot = np.array(points_to_plot)
        x, y = points_to_plot.T

        x = np.append(x, x[0])
        y = np.append(y, y[0])

        return x, y

    def validation(self):
        """
        Returns False for crossing shapes
        """
        points = self.vertices()
        shape = LinearRing(points)

        return shape.is_valid



def random_rotor_x_y():
    """
    Function which generates a rotor with random values for all variables
    Given the outer radius
    Given the center of shape
    It returns the x, y coordinates of the rotor to plot
    """

    inner_r = random.randint(1, 8)
    spikes = random.randint(5,15)
    angle = math.pi/(random.randint(2,18))

    rotor = Rotor(inner_r, spikes, angle)

    # If it is a bad rotor, return false
    if not rotor.validation():
        return False

    # Print description of rotor
    print(rotor.description())
    x,y = rotor.get_x_y()

    return x,y
