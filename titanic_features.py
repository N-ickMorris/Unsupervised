# -*- coding: utf-8 -*-
"""
Creates text features
Selects unique features using Hierarchical Clustering
Removes outliers

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.cluster import FeatureAgglomeration
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from plots import pairs_plot

# read in the data
data = pd.read_csv("titanic.csv")

# get the text columns form data
text = data[["Name", "Ticket"]]
data = data.drop(columns=["Name", "Ticket"])

# collect the words (and their inverse frequencies) from each document
# 'matrix' is a term (columns) document (rows) matrix
matrix = pd.DataFrame()
for c in text.columns:
    vector = TfidfVectorizer()
    matrix2 = vector.fit_transform(text[c].tolist())
    matrix2 = pd.DataFrame(matrix2.toarray(), columns=vector.get_feature_names())
    matrix = pd.concat([matrix, matrix2], axis=1)

# determine which columns are strings (for X)
x_columns = data.columns
x_dtypes = data.dtypes
x_str = np.where(x_dtypes == "object")[0]

# convert any string columns to binary columns
data = pd.get_dummies(data, columns=x_columns[x_str])

# fill in missing values with the (column) average
impute = SimpleImputer(strategy="mean")
columns = data.columns
data = pd.DataFrame(data=impute.fit_transform(data), columns=columns)

# separate X and Y
X = data.drop(columns=["PassengerId", "Survived"]).copy()
Y = data.drop(columns=X.columns).drop(columns="PassengerId").copy()
X_copy = X.copy()

# add the term-document matrix to X
X_copy = pd.concat([X_copy, matrix], axis=1)

# drop any constant columns in X
X_copy = X_copy.loc[:, (X_copy != X_copy.iloc[0]).any()]

# standardize the data to take on values between 0 and 1
X = ((X_copy - X_copy.min()) / (X_copy.max() - X_copy.min())).copy()

# build the feature selection model
num = 222
hclust = FeatureAgglomeration(n_clusters=num, linkage="ward", distance_threshold=None)
hclust.fit(X)

# collect the features to keep
clusters = hclust.labels_
keep = []
for i in range(num):
    keep.append(np.where(clusters == i)[0][0])
X_copy = X_copy.iloc[:, keep]

# standardize the data to take on values between 0 and 1
X = ((X_copy - X_copy.min()) / (X_copy.max() - X_copy.min())).copy()

# train a model to detect outliers
data = X.copy()
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
train_score = str(np.round(silhouette_score(X, components.loc[:, "Inlier"]), 3))

# plot the clusters
save_plot = False
pairs_plot(components, vars=components.columns[:n_comp],
           color="Inlier", title="Local Outlier Factor & PCA - Silhouette: " + train_score,
           save=save_plot)

# remove the outliers
good_idx = np.where(model.negative_outlier_factor_ > cutoff)[0]
X = X_copy.iloc[good_idx, :].reset_index(drop=True)
Y = Y.iloc[good_idx, :].reset_index(drop=True)

# drop any constant columns in X
X = X.loc[:, (X != X.iloc[0]).any()]

# export the data
X.to_csv("X titanic.csv", index=False)
Y.to_csv("Y titanic.csv", index=False)
