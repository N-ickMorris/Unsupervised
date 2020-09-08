# -*- coding: utf-8 -*-
"""
Creates features without any missing values

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from plots import pairs_plot


# read in the data
data = pd.read_csv("ansur.csv").drop(columns=["SUBJECT_NUMBER", "MONTH_MEASURED-ANSUR88", "YEAR_MEASURED-ANSUR88", "GRADE-ANSUR88", "RANK-ANSUR88", "MILITARY_PERSONNEL_CLASS", "BIRTH_PL_FATHER"])

# drop any constant columns in data
data = data.loc[:, (data != data.iloc[0]).any()]

# determine which columns are strings (for X)
x_columns = data.columns
x_dtypes = data.dtypes
x_str = np.where(x_dtypes == "object")[0]

# convert any string columns to binary columns
data = pd.get_dummies(data, columns=x_columns[x_str])

# fill in missing values with Bayesian Ridge Regression
impute = IterativeImputer(max_iter=5)
columns = data.columns
data = pd.DataFrame(data=impute.fit_transform(data), columns=columns)

# separate X and Y
X = data.drop(columns=["GENDER_Male", "GENDER_Female"]).copy()
Y = data.drop(columns=X.columns).drop(columns=["GENDER_Female"]).copy()

# train a model to detect outliers
data = pd.concat([Y, X], axis=1)
model = LocalOutlierFactor(n_neighbors=20, leaf_size=30, novelty=False, n_jobs=1)
model.fit(data)

# determine how much of the data has outliers
percent = 0.1
cutoff = np.quantile(model.negative_outlier_factor_, percent)
labels = (model.negative_outlier_factor_ > cutoff) * 1

# train a PCA model
n_comp = 3 # number of principal components
component = PCA(n_components=n_comp, random_state=42)
component.fit(X)

# compute components for all the data, add cluster labels and train/test labels
components = pd.DataFrame(component.transform(X), 
                          columns=["PC" + str(i + 1) for i in range(n_comp)])
components["Inlier"] = labels
components["Data"] = "Train"
components.to_csv("inliers and pca.csv", index=False)

# tells how well separated the clusters are
train_score = str(np.round(silhouette_score(X,
                                            components.loc[:, "Inlier"]), 3))

# plot the clusters
save_plot = False
pairs_plot(components, vars=components.columns[:n_comp],
           color="Inlier", title="Local Outlier Factor & PCA - Silhouette: " + train_score,
           save=save_plot)

# remove the outliers
good_idx = np.where(model.negative_outlier_factor_ > cutoff)[0]
X = X.iloc[good_idx, :].reset_index(drop=True)
Y = Y.iloc[good_idx, :].reset_index(drop=True)

# export the data
X.to_csv("X ansur.csv", index=False)
Y.to_csv("Y ansur.csv", index=False)
