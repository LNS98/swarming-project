"""
Plot the graphs for the allignment of paricle
"""

import matplotlib.pyplot as plt
import pandas as pd

def main():
    """
    Main program in function.
    """
    # list contining correct Ns
    N_list = [40, 100, 400]

    # loop around length of N
    for N in N_list:
        # import the data as dfs
        df = pd.read_csv("./Averages/N_{}.csv".format(N))

        # get the number of  averages
        num_averages = df["number of averages"].iloc[1]

        # plot the data
        plt.scatter(df["noise"], df["averages"], s = 2, label = "N = {} (repeats = {})".format(N, num_averages))
        plt.axis([0, 5.02, 0, 1.02])
        plt.xlabel("noise")
        plt.ylabel("allignment")
        plt.legend()
    plt.show()


main()
