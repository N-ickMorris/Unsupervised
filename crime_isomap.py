# -*- coding: utf-8 -*-
"""
Reduces the columns of a data set with an Isomap model

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestRegressor
from plots import scatter_plot


# read in the data
X = pd.read_csv("Y crimes.csv")

# standardize the data to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# separate the data into training and testing
np.random.seed(1)
test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# train a isomap model
n_comp = 1 # number of components
component = Isomap(n_components=n_comp, n_neighbors=5, n_jobs=1)
component.fit(X.iloc[train_idx, :])

# compute components for all the data, add cluster labels and train/test labels
components = pd.DataFrame(component.transform(X), 
                          columns=["IC" + str(i + 1) for i in range(n_comp)])
components["Data"] = "Train"
for j in test_idx:
    components.loc[j, "Data"] = "Test"
# components.to_csv("isomap.csv", index=False)

# combine the data and components
data = pd.concat([X, components], axis=1)

# train a random forest to learn the clusters
model = RandomForestRegressor(n_estimators=50, max_depth=10,
                              min_samples_leaf=5, max_features="sqrt",
                              random_state=42, n_jobs=1)
model.fit(X, components.drop(columns="Data"))

# collect and sort feature importance
importance = pd.DataFrame({"name": X.columns,
                           "importance": model.feature_importances_})
importance = importance.sort_values(by="importance", ascending=False).reset_index(drop=True)

# choose how many features to plot
num_features = 10
features = importance.loc[:(num_features - 1), "name"].tolist()

# plot the variables vs. components
comp_ = 0 # the column number of a component to plot
for c in features:
    scatter_plot(data=data,
                 x=c, y=components.columns[comp_],
                 color=None, title="Isomap",
                 legend=True, save=False)
