"""
This file initialised the population for the genetic algorithm
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random

from rotor_generator import Rotor, random_rotor
from simple_rotor import one_run


# pop = 20 # population number



rotor = random_rotor().fitness(plot = True)

print(rotor[-1])





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
