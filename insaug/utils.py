import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def mean_shift(data):
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    return labels, cluster_centers, n_clusters_


def compute_covariance_matrix(data):
    data = data - np.mean(data, axis=0)
    return np.cov(data, rowvar=False)


def compute_variance(data):
    return np.var(data, axis=0)