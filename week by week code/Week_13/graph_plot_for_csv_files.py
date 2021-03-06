"""
Plots all averages for the alignment of each system from the values written to the file in the previous program.
"""

import matplotlib.pyplot as plt
import pandas as pd

def main(type):
    """
    Main program in function.
    """
    # list contining correct Ns
    N_list = [100]
    markers = ["s"]


    # loop around length of N
    for N, m in zip(N_list, markers):
        # import the data as dfs
        df = pd.read_csv("./averages_{}/N_{}.csv".format(type, N))

        # get the number of  averages
        num_averages = df["number of averages"].iloc[1]

        # plot the data
        plt.scatter(df[type], df["averages"], s = 5, label = "N = {} (repeats = {})".format(N, num_averages), marker = m)
        # plt.axis([0, 5.02, 0, 1.02])
        plt.xlabel("noise")
        plt.ylabel("allignment")
        plt.legend()
    plt.show()


main("density")
