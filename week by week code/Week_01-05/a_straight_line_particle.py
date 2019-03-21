"""
Program with several functions producing random motion of a single particle and a particle in a straight line. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def straight_particle():
    """
    Functinon which just makes a moving particle.
    """
    # initialise time change step, 1 for now
    delta_t = 1

    # empty array which will contain the [v_x, x] coordinates of the particle
    positions = []

    # initiallise position and velocity of particle
    new_coord = [2, 0]

    # loop for certain amount of time and change position of particle
    while new_coord[1]  < 100:
        # update x
        new_coord = [2, new_coord[1] + new_coord[0] * delta_t]

        # append it to positions
        positions.append(new_coord)

    return positions


def random_motion():
    """
    Function which makes particle move in a random direction.
    """
    # initialise time change step, 1 for now
    delta_t = 1

    # empty array which will contain the [[x,y], [v_x, v_y]] coordinates of the particle
    positions = []
    velocities = []
    accelerations = []

    # initiallise position and velocity of particle
    pos = [0, 0]
    vel = [0, 0]

    # dELETE
    count = 0

    while count < 10:
        # get new values for x and y
        x_new = pos[0] + vel[0] * delta_t
        y_new = pos[1] + vel[1] * delta_t

        # get random values as the accelearion of the particles
        ax = random.randint(-1, 1)
        ay = random.randint(-1, 1)

        # get new values of velocity
        vx_new = ax * delta_t
        vy_new = ay * delta_t

        # update x and y positions and velocitis
        pos = [x_new, y_new]
        vel = [vx_new, vy_new]


        # append to positions
        positions.append(pos)
        velocities.append(vel)
        accelerations.append([ax, ay])

        #DELETE
        count += 1

    return positions, velocities, accelerations



def show_path(coordinates):
    """
    Function which takes in the coordinates as described in straight_particle and
    plots the result on a scatter graph.
    """

    # get the len of positions to use to get time coordinates
    number_steps = len(coordinates)

    # create time array to then plot
    time_data = [t for t in range(number_steps)]

    # get x and y coordiantes
    x_coor = []
    y_coor = []
    for pos in coordinates:
        x_coor.append(pos[0])
        y_coor.append(pos[1])




    # # plot time vs distance graph
    # plt.scatter(time_data, x_coor)
    # plt.title("distance vs time graph for a particle moving in a striaght line")
    # plt.xlabel("time")
    # plt.ylabel("x position")
    # #plt.show()


    # plot x, y distance graph
    plt.plot(x_coor, y_coor)
    plt.title("distance vs time graph for a particle moving in a striaght line")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.show()


# # run functions
# print("positions: {}".format(random_motion()[0]))
# print("velocities: {}".format(random_motion()[1]))


show_path(random_motion()[2])
show_path(random_motion()[1])
show_path(straight_particle())
show_path(random_motion()[0])
