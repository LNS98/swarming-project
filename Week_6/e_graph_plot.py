"""
Plots all averages for the alignment of each system from the values written to the file in the previous program. 
"""

import matplotlib.pyplot as plt
import pandas as pd

def main():
    """
    Main program in function.
    """
    # list contining correct Ns
    N_list = [40, 100, 400]
    markers = ["s", "v", "o"]


    # loop around length of N
    for N, m in zip(N_list, markers):
        # import the data as dfs
        df = pd.read_csv("./Averages/N_{}.csv".format(N))

        # get the number of  averages
        num_averages = df["number of averages"].iloc[1]

        # plot the data
        plt.scatter(df["noise"], df["averages"], s = 5, label = "N = {} (repeats = {})".format(N, num_averages), marker = m)
        plt.axis([0, 5.02, 0, 1.02])
        plt.xlabel("noise")
        plt.ylabel("allignment")
        plt.legend()
    plt.show()


main()
