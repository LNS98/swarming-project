"""
This file initialises the population for the genetic algorithm
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random

from rotor_generator import *


pop = 20 # population number


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
        # print(self.container)


        for i in range(len(self.container)):
            element = self.container[i]
            print(element)
            theta_array = element.fitness(plot = False)
            scores[element] = theta_array[-1]

        return scores


population = Population(4)
population.initialise()

print(population.calc_fitness())







#
# rotor = random_rotor().fitness(plot = True)
#
# print(rotor[-1])





# for i in range(pop):
#     rotor = random_rotor_x_y()
#
#     # skip the rotors which are wrong
#     if rotor == False:
#         continue
#
#     x, y = rotor
#
#
#     plt.plot(x,y)
#     plt.show()



#
# x, y = Rotor(8, 5, 0.19634954084936207).get_x_y()
#
# plt.plot(x, y)
# plt.show()


#
# 1.5707963267948966,
# 10 spikes
# 8 inner rad
