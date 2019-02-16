"""
Code trying to get a mating pool for the genetic algorithm
"""
import matplotlib.pyplot as plt
import numpy as np
import random



def mating_pool(given_dict):
    """
    Return an array containg the correct number of relative amounts of the id's
    given their fitness levels.
    """
    keys = given_dict.keys()
    vals = given_dict.values()

    # change the values of the fitness score to be positive intigers
    vals = [int(round(abs(i * 1000))) for i in val]

    dict = {i:j for i, j in zip(keys, vals)}
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

        # mutate child
        # child = mutation(child)

        # place the child in the pop
        pop[i] = child

    return pop
