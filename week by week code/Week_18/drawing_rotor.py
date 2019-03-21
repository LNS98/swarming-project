"""
Program to help explain how the rotors are built.
"""
import math
import matplotlib.pyplot as plt
import numpy  as np

L = 10
b = L * 0.4

def main():
    poly = polygon([L/2, L/2], L * 0.3, math.pi/3, 8)

    poly = np.array(poly)
    x, y = poly.T

    x = np.append(x, x[0]).tolist()
    y = np.append(y, y[0]).tolist()

    plt.plot(x, y, "k-")
    plt.scatter(x, y, s = 20, c = "r")
    plt.show()


    return 0

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

    # plot inner
    pts_for_plot_out = np.array(pts_out)
    x_out, y_out = pts_for_plot_out.T

    plt.axis([0, L, 0, L])
    plt.scatter(x_out, y_out, s = 20, c = "r")
    plt.show()

    # plot the outer points
    pts_for_plot_in = np.array(pts_in)
    x_in, y_in = pts_for_plot_in.T

    plt.axis([0, L, 0, L])
    plt.scatter(x_out, y_out, s = 20, c = "r")
    plt.scatter(x_in, y_in, s = 20, c = "r")
    plt.show()


    sorted_out = sorted(pts_out, key=clockwiseangle_and_distance)
    sorted_in = sorted(pts_in, key=clockwiseangle_and_distance)

    for i in range(len(sorted_in)):
        pts_tot.append(sorted_out[i])
        pts_tot.append(sorted_in[i])

    return pts_tot




main()
