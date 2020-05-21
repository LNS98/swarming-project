"""
This file initialises the population for the genetic algorithm
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool

from rotor_generator import *

generations = 30

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


    def pop_from_csv(self, path_to_file):
        """
        Takes the population from a csv file and makes it a population object.
        """

        df = pd.read_csv(path_to_file)

        for index, row in df.iterrows():
            angle = row["angle"]
            spikes = int(row["spikes"])
            inner_r = row["inner rad"]
            fitness = row["fitness"]

            rotor = Rotor(inner_r, spikes, angle)
            rotor.fit = fitness
            self.container.append(rotor)

        return None

    def calc_fitness(self):
        """
        When this method is called, it calculates the fitness of each element
        in the population and returns a dictionary with all of the elements
        and their respective score (fitness score)
        """
        scores = {}
        data = []


        for rotor in self.container:

            # create a pool with 24 processes for 24 averages for each rotor fitness
            p = Pool(processes = 24)

            # get the fitness scores simulatneously for each rotor
            data_list = p.map(fitness_mlp, [rotor for i in range(24*4)])

            # wait till the processes are finished
            p.close()
            p.join()

            # take average of score
            data_list = np.array(data_list)
            data_value = np.mean(data_list)

            # add this value to list of data values
            data.append(data_value)

        for i in range(len(self.container)):
            # create a ditionary witht the rotors as keys and fitness scores as
            element = self.container[i]
            scores[element] = data[i]

            # set the fit property of the given rotar to the new fitness score
            self.container[i].fit = data[i]

        return scores

    def mating_pool_fun(self, fit_scores):
        """
        Return an array containg the correct number of relative amounts of the rotors
        based on their fitness levels.
        Feed in the fitness scores from the calf_fitness function.
        """
        given_dict = fit_scores

        keys = list(given_dict.keys())
        vals = list(given_dict.values())


        # change the values of the fitness score to be positive integers
        for ind, i in enumerate(vals):
            if i > 0:
                vals[ind] = int(round(i * 1000))
            else:
                vals[ind] = 0

        dict = {i:j for i, j in zip(keys, vals)}

        mat_pool = [x for x in dict for y in range(dict[x])]

        return mat_pool

    def mutation(self, rotor):
        """
        Change a child by a given mutation rate.
        """
        mu_rate = 0.01
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

        return rotor

    def generate(self, mating_pool):
        """
        Generate next population.
        Feed in the mating  pool from the returned value of the mating pool
        """


        for i in range(self.member_number):
            # choose two random parents
            parA = random.choice(mating_pool)
            parB = random.choice(mating_pool)

            # print("\n Parents")
            #
            # parA.description()
            # parB.description()


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
            child = self.mutation(child)

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
        scores = [i.fit for i in self.container]

        average = np.mean(np.array(scores)) # mean
        std = np.std(np.array(scores))      # standard deviation
        st_e = std/(math.sqrt(len(scores))) # standard error

        print("average: {}".format(average))
        return average, std, st_e



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
        new_angle = angle_over_t[-1] + ang_velocities_obj[0] * delta_t

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
    # lists for outputting to the files
    average_list = []
    std_list = []
    ste_list = []


    population = Population(30)

    # population.initialise()

    population.pop_from_csv("./population_old.csv")


    for i in range(generations):

        if i != 0:
            # calculate the fitness of the rotars. Change both their fit values and return a dictionary with
            # the rotor and the corresponding fitness value called scores
            scores = population.calc_fitness()

        else:
            scores = {}
            for i in range(len(population.container)):
                # create a ditionary witht the rotors as keys and fitness scores as
                element = population.container[i]
                scores[element] = element.fit

        # # print out the population for debuging reasons
        population.describe_pop()

        # calculate the average fitness
        stat_values = population.average_fitness()
        average_list.append(stat_values[0])
        std_list.append(stat_values[1])
        ste_list.append(stat_values[2])

        # lists and operation to write the popultation to a file
        outer_r_list = []
        inner_r_list = []
        angle_list = []
        spikes_list = []
        fitness_list = []
        for roti in population.container:
            outer_r_list.append(roti.outer_r)
            inner_r_list.append(roti.inner_r)
            angle_list.append(roti.angle)
            spikes_list.append(roti.spikes)
            fitness_list.append(roti.fit)


        df1 = pd.DataFrame({"outer rad": outer_r_list, "inner rad": inner_r_list,
        "angle": angle_list, "spikes": spikes_list, "fitness": fitness_list})
        df2 = pd.DataFrame({"average": average_list, "std": std_list, "ste": ste_list})


        df1.to_csv("./population.csv", mode='w')

        df2.to_csv("./trials.csv")

        # calculate the mating pool from which we will sample parents for the new gen of children
        mat_pool = population.mating_pool_fun(scores)

        # generate the new population of children, hence the new populaion
        population.generate(mat_pool)

        print("------------------------- Time Taken: {} -------------------".format(time.time() - start))


    return None

def help():

    population = Population(30)

    population.pop_from_csv("./Results/Pop_30_no_align_24x4repeats/population.csv")

    population.describe_pop()

    return None

if __name__ == "__main__":
    start = time.time()
    main()
    # help()
