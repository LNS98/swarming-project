"""
This is a program to merge the data received from both computers when running the code for noise vs alignment
"""

import matplotlib.pyplot as plt
import pandas as pd




def main():

    # Read in both files
    lor = pd.read_csv("./Averages/N_100.csv")
    yon = pd.read_csv("./Averages/N_100_yonni.csv")

    print(lor)

    lor["averages"] = (lor["number of averages"] * lor["averages"] + yon["averages"] *  yon["number of averages"]) \
    / (yon["number of averages"] + lor["number of averages"])

    lor["number of averages"] = (yon["number of averages"] + lor["number of averages"])


    lor.to_csv("./N_100_tot.csv")

    return None

main()
