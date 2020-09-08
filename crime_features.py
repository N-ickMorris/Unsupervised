# -*- coding: utf-8 -*-
"""
Creates 2nd order polynomial features
Selects best features using Random Forest

@author: Nick
"""


import numpy as np
import pandas as pd


# read in the data
data = pd.read_csv("crimes.csv")

# determine which columns are strings (for X)
x_columns = data.columns
x_dtypes = data.dtypes
x_str = np.where(x_dtypes == "object")[0]

# convert any string columns to binary columns
data = pd.get_dummies(data, columns=x_columns[x_str])

# fill in missing values
data = data.fillna(method="bfill").fillna(method="ffill")

# separate X and Y
X = data.drop(columns=["YEAR.WEEK", "North CRIMES", "West CRIMES", "Central CRIMES"]).copy()
Y = data.drop(columns=X.columns).drop(columns="YEAR.WEEK").copy()

# export the data
X.to_csv("X crimes.csv", index=False)
Y.to_csv("Y crimes.csv", index=False)
