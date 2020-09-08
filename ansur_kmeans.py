# -*- coding: utf-8 -*-
"""
Trains and tests a k-Means clustering model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from plots import pairs_plot, corr_plot


# read in the data
X = pd.read_csv("X ansur.csv")

# standardize the data to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# plot correlations to see how many clusters there are
corr_plot(X, method="complete")

# separate the data into training and testing
np.random.seed(1)
test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# train a k-means model
cluster = KMeans(n_clusters=4, n_init=20, max_iter=300, tol=0.0001, random_state=42,
                 n_jobs=1)
cluster.fit(X.iloc[train_idx, :])

# compute clusters for all the data
labels = cluster.predict(X)

# train a PCA model
n_comp = 3 # number of principal components
component = PCA(n_components=n_comp, random_state=42)
component.fit(X.iloc[train_idx, :])

# compute components for all the data, add cluster labels and train/test labels
components = pd.DataFrame(component.transform(X), 
                          columns=["PC" + str(i + 1) for i in range(n_comp)])
components["Cluster"] = labels
components["Data"] = "Train"
for j in test_idx:
    components.loc[j, "Data"] = "Test"
components.to_csv("kmeans and pca.csv", index=False)

# tells how well separated the clusters are
train_score = str(np.round(silhouette_score(X.iloc[train_idx, :],
                                            components.loc[train_idx, "Cluster"]), 3))
test_score = str(np.round(silhouette_score(X.iloc[test_idx, :],
                                           components.loc[test_idx, "Cluster"]), 3))

# plot the clusters
save_plot = False
pairs_plot(components, vars=components.columns[:n_comp],
           color="Cluster", title="K-Means Clustering & PCA - Silhouette: " + train_score,
           save=save_plot)

# train a random forest to learn the clusters
model = RandomForestClassifier(n_estimators=50, max_depth=10,
                               min_samples_leaf=5, max_features="sqrt",
                               class_weight="balanced_subsample",
                               random_state=42, n_jobs=1)
model.fit(X, labels)

# collect and sort feature importance
importance = pd.DataFrame({"name": X.columns,
                           "importance": model.feature_importances_})
importance = importance.sort_values(by="importance", ascending=False).reset_index(drop=True)

# choose how many features to plot
num_features = 3
df = pd.concat([X, pd.DataFrame(labels, columns=["cluster"])], axis=1)
features = importance.loc[:(num_features - 1), "name"].tolist()

# plot the variables
pairs_plot(df, vars=features,
           color="cluster", title="K-Means Clustering - Silhouette: " + train_score,
           save=save_plot)
