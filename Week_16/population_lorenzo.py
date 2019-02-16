"""
This file initialises the population for the genetic algorithm
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool

from rotor_generator import *



class Population:

    # Instances are specific to each object
    def __init__(self, member_number):
        self.member_number = member_number
        self.container = []

    def initialise(self):
        """
        if this method is called, it will initialise the population
        with size member_number
        """

        for i in range(self.member_number):
            self.container.append(random_rotor())

    def calc_fitness(self):
        """
        When this method is called, it calculates the fitness of each element
        in the population and returns a dictionary with all of the elements
        and their respective score (fitness score)
        """
        scores = {}

        # create a pool with 20 processes
        p = Pool(processes = 24)

        # get the fitness scores simulatneously for each rotor
        data = p.map(fitness_mlp, self.container)

        # wait till the processes are finished
        p.close()
        p.join()

        for i in range(len(self.container)):
            element = self.container[i]
            scores[element] = data[i]

        return scores

    def mating_pool_fun(self):
        """
        Return an array containg the correct number of relative amounts of the rotors
        based on their fitness levels.
        """
        given_dict = self.calc_fitness()

        keys = list(given_dict.keys())
        vals = list(given_dict.values())


        # change the values of the fitness score to be positive integers
        for ind, i in enumerate(vals):
            if i > 0:
                vals[ind] = int(round(i * 1000))
            else:
                vals[ind] = 0

        dict = {i:j for i, j in zip(keys, vals)}

        c = [x for x in dict for y in range(dict[x])]

        return c

    def mutation(self, rotor):
        """
        Change a child by a given mutation rate.
        """

        repeat = True

        while repeat:
            properties = [rotor.inner_r, rotor.spikes, rotor.angle]

            for element in properties:
                # generate a random number
                num = random.random()
                if num < mu_rate:
                    #change element of to a random choice
                    rotor.mutate(element)

            if rotor.validation():
                repeat = False

        return None

    def generate(self):
        """
        Generate next population.
        """

        mating_pool = self.mating_pool_fun()

        for i in range(self.member_number):
            # choose two random parents
            parA = random.choice(mating_pool)
            parB = random.choice(mating_pool)

            # get the values of inner_r, angles, spikes
            inner_r = random.choice([parA.inner_r, parB.inner_r])
            spikes = random.choice([parA.spikes, parB.spikes])
            angle = random.choice([parA.angle, parB.angle])

            # make a child out of these values
            child = Rotor(inner_r, spikes, angle)

            # If it is a bad rotor, return false
            if not child.validation():
                child = random_rotor()

            # mutate child
            # child = mutation(child)

            # place the child in the pop
            self.container[i] = child

        return None

    def describe_pop(self):
        """
        Returns the description of elements in population
        """

        for element in self.container:
            element.description()

        return None

    def average_fitness(self):
        """
        This prints the average fitness value
        """
        scores = self.calc_fitness()

        vals = np.array(list(scores.values()))
        print("Average fitness: {}".format(np.mean(vals)))

        return vals








def fitness_mlp(rotor):
    """
    returns the final angle which the rotor has moved for a
    set amount of time T_final
    """

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


def main():
    # get a function which contains the lsit of average values
    average_list = []

    population = Population(24)
    population.initialise()
    for i in range(10):
        population.calc_fitness()
        population.generate()
        average_list.append(population.average_fitness())
        print("------------------------- Time Taken: {} -------------------".format(time.time() - start))

    plt.plot([i for i in range(10)], average_list)
    plt.show()
    return None


def help():
    rotor = random_rotor()

    value = rotor.fitness()
    print(value)

    return None


if __name__ == "__main__":
    start = time.time()
    main()
    # help()
