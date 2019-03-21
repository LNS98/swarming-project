
import matplotlib.pyplot as plt
import pandas as pd
from simple_rotor_contact_force import *
from rotor_generator import *




def main():

    # loop through all teh files
    for i in range(4):
        # get the data
        x_values, y_values = get_data("./genetic_algorithm/alignment/noise_2/population_{}_n_2.csv".format(i + 1))
        # plot the population
        plots(x_values, y_values)

    return 0


def get_data(filename):
    """
    Get the data for all the plots from a given pop folder
    """
    # get the data as a df
    df = pd.read_csv(filename)

    # mak an empty list which will contain the x and y lists
    x_list = []
    y_list = []

    # loop over each rotor
    for i in df.iterrows():
        # build the rotor
        inner_r = i[1]["inner rad"]
        spikes = int(i[1]["spikes"])
        angle = i[1]["angle"]

        rot = Rotor(inner_r, spikes, angle)

        # get the x and y values and append  them to the bigger list
        x, y = rot.get_x_y()

        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


def plots(x_s, y_s):
    """
    Plot the population of  rotors in a subplot configuration.
    """
    # get the figure with the amount of subplots
    fig, ax = plt.subplots(nrows=5, ncols=6, squeeze = False)

    # variable to loop through the x_s and y_s
    count = 0
    # loop through each row  and column to populate graph
    for row in ax:
        for col in row:
            # plot the figure
            col.plot(x_s[count], y_s[count])
            col.axis('off')

            # add 1  to the count
            count += 1

    plt.show()

    return None



main()
