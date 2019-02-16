"""
Code trying to get a mating pool for the genetic algorithm
"""
import matplotlib.pyplot as plt
import numpy as np
import random



def main():

    #  test id's and fitness values to put in the dictionary
    ids = ['{}'.format(i) for i in range(10)]
    fitness = [int(abs(round(random.random() * 100))) for i in range(10)]

    # place the values in a dictionary
    test_dict = {i:j for i, j in zip(ids, fitness)}

    # call the maiting function
    mat_pool = mating_pool(test_dict)

    return 0

def mating_pool(given_dict):
    """
    Return an array containg the correct number of relative amounts of the id's
    given their fitness levels.
    """
    c = [x for x in given_dict for y in range(given_dict[x])]

    return c

def generate(pop, maiting_pool):
    """

    """
    N = len(pop)


    for i in range(N):
        # choose two random parents
        parA = random.choice(maiting_pool)
        parB = random.choice(maiting_pool)

        # get the values of inner_r, angles, spikes
        inner_r = random.choice([parA.inner_r, parB.inner_r])
        spikes = random.choice([parA.spikes, parB.spikes])
        angle = random.choice([parA.angle, parB.angle])

        # make a child out of these values
        child = Rotor(inner_r, spikes, angle)

        # place the child in the pop
        pop[i] = child

    return pop


def mutation(mut_rate, child):
    """
    Change a child by a given mutation rate.
    """

    for element in child:
        # generate a random number
        num = random.random()
        if num < mu_rate:
            #change element of to a random choice
            child.element = random_element(element)



    return None
