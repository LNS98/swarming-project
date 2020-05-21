"""
Try to find how the alignment and rotation rate depend on each other by  implementing OLS
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    # load the data
    df  = import_data()

    # get a series of points for x and f(x)
    x_list = np.linspace(2, 20, num = 1000)
    y_list = [loss_function(df, x) for x in x_list]

    # get the minimum value of y
    y_min = min(y_list)
    x_min = x_list[y_list.index(y_min)]

    print(x_min, y_min)


    plt.plot(x_list, y_list)
    plt.show()

    # raise alignment to the x version
    df["alignment_power"] = df["alignment"] ** x_min

    # propagate the error
    df["alignment_power_error"] = abs((df["alignment"] + df["alignment_error"]) ** x_min -  df["alignment"] ** x_min)

    # send to a file to compare the results
    # df.to_csv("./ali_dependence_and_omega.csv", index = False)
    #
    # # plot the result
    # plt.scatter(df["noise"], df["alignment_power"], s = 2)
    # plt.scatter(df["noise"], df["omega"], s = 2)
    # plt.show()

    return 0


def import_data():
    """
    Import the data from the two files and store them in pandas df.
    """

    # import the data for w
    df_w = pd.read_csv("./alignment/align_noise_variation.csv")
    # import the data for alignment
    df_a = pd.read_csv("C:/Users/lorni/OneDrive/Documenti/Physics/Year_3/Physcis_project/swarming-project/results/kNN_model/averages_N_40.csv")


    # rescale the w values to be between 0 and 1
    df_w["average_scaled"] = (df_w["average"] - df_w["average"].min()) / (df_w["average"].max() - df_w["average"].min())

    # propagate the error

    df_w["mult_factor"] = df_w["average_scaled"] / df_w["average"]
    df_w["error_on_average_scaled"] = abs(df_w["mult_factor"]) * df_w["std error"]

    # find the error of the min and max average
    #a.c1[a.c1 == 8].index.tolist()
    # error_max = float(df_w["std error"].iloc[df_w.average[df_w.average == df_w["average"].max()].index.tolist()])
    # error_min = float(df_w["std error"].iloc[df_w.average[df_w.average == df_w["average"].min()].index.tolist()])
    #
    #
    #
    # df_w["error_on_average_scaled"] = abs(((df_w["average"] + df_w["std error"]) - (df_w["average"].min() + error_min)) /
    #                                   ((df_w["average"].max() + error_max) - (df_w["average"].min() + error_min)) - df_w["average_scaled"])



    print(df_w)

    # place them in a new df containing nosie, w, al
    df = pd.DataFrame(data = {"noise": df_a["noise"], "omega": df_w["average_scaled"], "omega_error": df_w["error_on_average_scaled"], "alignment": df_a["mean (101 averages)"], "alignment_error": df_a["std_error"]})

    return df

def loss_function(df, x):
    N = len(df["noise"])

    result = 0
    # loop to sum over all i's
    for i in range(N):
        argument = (df["alignment"].loc[i] ** (x) - df["omega"].loc[i]) ** 2
        result += argument

    return result

def deriv_function(df, x):
    """
    calculates the loss function to minimise which varies with x.
    df gives the values for the sum and x is the variable.
    """
    N = len(df["noise"])

    result = 0
    # loop to sum over all i's
    for i in range(N):
        argument = 2 * x * df["alignment"].loc[i] ** (x - 1) *  (df["alignment"].loc[i] ** (x) - df["omega"].loc[i])
        result += argument

    return result


main()
