"""
Envvironement that will contain the simulation.
"""

#from utils import centroid, rescale
#from constants import N, v_mag, L, delta_t, dimensions

from rotors import Rotor

import numpy as np
import random
import cv2

import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.cm as cm


class Environment:
    # some constants
    M = 1   # number of objects
    V_MAG = 0.05      # total magnitude of each particle velocity
    DELTA_T = 1     # time increment

    # # distance metrics in the code
    R = 1.0   # radius of allignment
    R_C = 0.05 # radius within repulsion
    R_E = 0.5 # radius of equilibrium between the particles
    R_A = 0.8 # radius when attraction starts
    R_O = 0.05 # radius of attraction between the particels and the objects
    DIMENSIONS = 2   # dimensions
    TIME_PAUSE = 1e-2 # time pause for interactive graph
    PERIODIC_BOUNDARIES = True

    def __init__(self, L, N, mag=200):
        
        self.mag = mag
        self.L = L # height of box
        self.N = N # number of particles
       
        # image that contains simulation 
        self.image = np.zeros([int(self.mag*self.L),
                               int(self.mag*self.L),
                               3], dtype=np.uint8)
        
        self._pop_box()
        # forces parameters
        # alpha = 0 # stregnth of repulsive force between to the particles
        # beta = 1 # stregnth of the force due to the objects on the particles
        # gamma = 1 # stregnth of allignment
        # fric_force = 0.2  # frictional force of the object when rotating
        # noise = 1  # noise added to the velocity

        # model type (SVM, kNN)

        pass

    def _pop_box(self):
        # self.agents = []
        
        # create rotor at centre of box

        # for number of partciles
            # get x y coords - if not in rotor, init agent

        self.rotor = Rotor((0.5*self.L, 0.5*self.L), 15, self.L*0.1, self.L*0.2, np.pi/2)

    def step(self):
        # for number of particles

            # agent.update()

        # update the rotor
        pass

    def display(self):
        
        self.image.fill(0)

        # draw rotor
        for i in range(len(self.rotor.verticies)):
            # get the positions of the start and end of the lines 
            start = (int(self.mag*self.rotor.verticies[i][0]), int(self.mag*self.rotor.verticies[i][1]))
            end = (int(self.mag*self.rotor.verticies[(i+1)%len(self.rotor.verticies)][0]), int(self.mag*self.rotor.verticies[(i+1)%len(self.rotor.verticies)][1]))
            cv2.line(self.image, start, end, (0, 20, 200), 1)

        cv2.imshow('Simulation', self.image)
        cv2.waitKey(0)



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


# if __name__ == "__main__":
#     start = time.time()
#     main()
#     # help()
#     print("------------------------- Time Taken: {} -------------------".format(time.time() - start))
