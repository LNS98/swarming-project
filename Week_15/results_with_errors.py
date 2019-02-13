"""
Takes the csv file (for given N) containg the repeats for each noise level,
computes, average, std, standard error and exports to new file.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(type, model, N, folder, name_of_file):

    # read in the data as a df from the correct location
    df = pd.read_csv("./time_averages_{}_{}/N_{}_U_20000.csv".format(type, model, N))

    # get the number of averages
    num_averages = len(df.columns) - 1


    # get the mean,  std, standard error
    df["mean ({} averages)".format(num_averages)] = df.drop(type, 1).mean(axis = 1)
    df["std"] = df.drop(["mean ({} averages)".format(num_averages), type], 1).std(axis = 1)
    df["std_error"] = df.drop([type, "mean ({} averages)".format(num_averages), "std"], 1).std(axis = 1) / np.sqrt(num_averages)

    # get another df with  the correct information and write it to a results file
    df2 = df[[type, "mean ({} averages)".format(num_averages), "std", "std_error"]]
    df2.to_csv("../results/{}/{}.csv".format(folder, name_of_file), index = False)

    return 0


main("noise", "SVM", 40, "SVM", "noise_time_averages_U_20000_N40")
