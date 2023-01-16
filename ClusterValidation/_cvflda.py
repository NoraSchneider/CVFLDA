"""Cluster validation method using Fishers's LDA-direction of projection"""

import numpy as np
import math
import random
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing, check_X_y
from scipy.stats import t
import warnings


class CVFLDA:
    """
    Cluster Validation

    X: ndarray, the data matrix.
    y: ndarray, the original input labels of the clustering for X
    safety_margin: int, default = 2, margin of safety for decision
    sequential: bool, default = True, sequential or simultaneous merging.

    adjusted_y: ndarray, the adjusted cluster labels after validating.
    """

    def __init__(self, X, y, safety_margin=2, sequential=True):
        # Check if X and labels are in the right format (right dimensions, sizes etc.)
        self.original_labels = y
        self.X, y = check_X_y(X, y)

        # Transform Labels to numerical labels from 0 to k (= number of Clusters)
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(y)
        self.adjusted_y = None

        self.safety_margin = safety_margin
        self.sequential = sequential

    def validate(self):
        """
        This function estimates the optimal number of clusters based on a given clustering
        """

        adjusted_y = self.y
        if len(self.le.classes_) == 1:
            warnings.warn("Only one cluster in provided clustering.")
            self.adjusted_y = adjusted_y
            return adjusted_y

        # A cluster has to contain at least 2 objects
        label_counts = np.unique(adjusted_y, return_counts=True)

        if 1 in label_counts[1]:
            adjusted_y = self._relocate_single_point_cluster(adjusted_y, label_counts)

        matrix_comparison = self._compare_clusters(adjusted_y)
        merge_index = np.transpose(np.nonzero(matrix_comparison))

        # as long as the pair-wise comparison of clusters show that there are false-cluster pairs, they are merged until
        # the clustering consists of only true clusters
        if self.sequential:
            while merge_index.any():

                nonzero = matrix_comparison[np.nonzero(matrix_comparison)]
                index_min = np.where(matrix_comparison == np.min(nonzero[np.logical_not(np.isnan(nonzero))]))
                index_min = np.concatenate(index_min)
                # print("Matrix Comparison", matrix_comparison, "inmin", index_min)
                adjusted_y = np.where(adjusted_y == index_min[1], index_min[0], adjusted_y)
                adjusted_y = self.le.fit_transform(adjusted_y)

                if len(self.le.classes_) == 1:
                    break
                else:
                    matrix_comparison = self._compare_clusters(adjusted_y)
                    merge_index = np.transpose(np.nonzero(matrix_comparison))
        else:

            while merge_index.any():
                g = nx.Graph(merge_index.tolist())
                for subgraph in nx.connected_components(g):
                    for cluster in subgraph:
                        adjusted_y = np.where(adjusted_y == cluster, list(subgraph)[0], adjusted_y)

                adjusted_y = self.le.fit_transform(adjusted_y)
                if len(self.le.classes_) == 1:
                    break
                else:
                    matrix_comparison = self._compare_clusters(adjusted_y)
                    merge_index = np.transpose(np.nonzero(matrix_comparison))

        self.adjusted_y = adjusted_y
        return self

    def get_adjusted_cluster_count(self) -> int:
        """
        Gets the number of distinct clusters in adjusted clustering
        :return: int, number of distinct clusters in adjusted clustering
        """
        if self.adjusted_y is None:
            self.validate()
        return len(np.unique(self.adjusted_y))

    def _relocate_single_point_cluster(self, adjusted_y, label_counts):
        """
        When there are clusters with one single point, it allocates single points to nearest cluster and returns
        an array of adjusted labels
        :param adjusted_y: current y labels
        :param label_counts: list, with cluster names and their frequency
        :return: adjusted cluster labels
        """
        le = LabelEncoder()
        means = self._calculate_cluster_means(adjusted_y, label_counts)
        while 1 in label_counts[1]:
            for i in range(len(label_counts[0])):
                if label_counts[1][i] == 1:
                    mean_i = means[i]
                    for j in range(len(means)):
                        dist = np.linalg.norm(means[j] - mean_i)
                        if dist == 0:
                            continue
                        elif j == 0:
                            min_dist = dist
                            cluster_index = j
                        elif j == 1 and i == 0:
                            min_dist = dist
                            cluster_index = j
                        elif dist < min_dist:
                            min_dist = dist
                            cluster_index = j
                    adjusted_y = np.where(adjusted_y == i, cluster_index, adjusted_y)
                    adjusted_y = le.fit_transform(adjusted_y)
                    label_counts = np.unique(adjusted_y, return_counts=True)
                    means = self._calculate_cluster_means(adjusted_y, label_counts)
                    break

        return adjusted_y

    def _calculate_cluster_means(self, y , label_counts):
        means = []
        for i in label_counts[0]:
            cluster_i = _safe_indexing(self.X, y == i)
            mean_i = cluster_i.mean(axis=0)
            means.append(mean_i)
        return means

    def _compare_clusters(self, labels):
        """
        This function evaluates a clustering for true and false cluster pairs
        :param labels:  array-like of shape (n_samples,)
                        cluster-labels for each object
        :return: matrix_comparison array-like of shape (k,k)
                pair-wise comparison matrix, where the entries are 0 if the clusters are true clusters and non-zero for
                false clusters
        """

        n_labels = len(np.unique(labels))
        result = np.zeros((n_labels, n_labels))

        # pairwise comparison of clusters
        for i in range(n_labels):
            cluster_i = _safe_indexing(self.X, labels == i)
            n_i = len(cluster_i)

            for j in range(i + 1, n_labels):
                cluster_j = _safe_indexing(self.X, labels == j)
                n_j = len(cluster_j)

                u = _projection_direction(cluster_i, cluster_j)

                var_i, var_var_i = _variance_in_direction(cluster_i, u)
                var_j, var_var_j = _variance_in_direction(cluster_j, u)

                # merged cluster based on direction
                cluster_merged = _merged_cluster(cluster_i, cluster_j, u)
                var_merged, var_var_merged = _variance_in_direction(cluster_merged, u)
                n_merged = len(cluster_merged)

                testresult = _compare_variances(var_i, var_var_i,  var_j, var_var_j, self.safety_margin)

                if not testresult:
                    result[i, j] = (2 * var_merged) / (var_i + var_j)

        return result


def _projection_direction(c_i, c_j):
    """ This function calculates the projection direction according to the Fisher criterion

        :param c_i: array-like of shape (n_i, n_features)
        :param c_j: array-like of shape (n_j, n_features)
        :return: direction: array-like of shape (n_features,), Direction = SW^-1 * (mu_i-mu_j)
    """

    """
        if len(c_i) == 1:
            cov_i = np.zeros((len(c_i[0]), len(c_i[0])))
        else:
            cov_i = np.cov(c_i.T)
        if len(c_j) == 1:
            cov_j = np.zeros((len(c_j[0]), len(c_j[0])))
        else:
            cov_j = np.cov(c_j.T)
        """
    cov_i = np.cov(c_i.T)
    cov_j = np.cov(c_j.T)

    sw = (len(c_i) - 1) * cov_i + (len(c_j) - 1) * cov_j
    sw = sw + np.diag(np.full(sw.shape[0], 0.0001))

    sw_inv = np.linalg.inv(sw)

    mean = np.mean(np.concatenate((c_i, c_j), axis=0), axis=0)

    c_i_deviation = c_i.mean(axis=0) - mean
    c_j_deviation = c_j.mean(axis=0) - mean
    sb = len(c_i) * np.matmul(c_i_deviation[:, None], c_i_deviation[None, :]) + len(c_j) * np.matmul(
        c_j_deviation[:, None],
        c_j_deviation[None, :])
    A = np.matmul(sw_inv, sb)

    W, V = np.linalg.eig(A)
    index_max_eigenvalue = np.argmax(W)
    direction_old = V[:, index_max_eigenvalue]

    direction = np.zeros(direction_old.shape)
    for i in range(len(direction_old)):
        if (direction_old[i].imag == 0):
            direction[i] = direction_old[i].real

    return direction


def _variance_in_direction(c, direction):
    """ This function calculates the variance of c along the specified direction and the variance of this variance
    :param c: array-like of shape (n_c, n_features)
    :param direction: array-like of shape(n_feature,)
    :return: variance: float
            var_of_var: float
    """
    mean = c.mean(axis=0)
    c = c - mean
    projected_points = np.matmul(mean, direction) + np.matmul(c, direction)
    variance = np.var(projected_points)
    standardized = projected_points - np.mean(projected_points)
    standardized_squared = np.multiply(standardized, standardized)
    var_of_var = np.var(standardized_squared) / len(c)

    return variance, var_of_var


def _merged_cluster(cluster_i, cluster_j, direction):
    """ This function calculates the merged Cluster, based on the half of projected objects in cluster i,
    that are closest to the projected mean of cluster j (and the other way around)

    :param cluster_i: array-like of shape (n_i, n_features)
            Cluster i
    :param cluster_j: array-like of shape (n_j, n_features)
            Cluster j
    :param direction: array-like of shape (n_features,)
            Direction, on which the merged cluster is based
    :return: c_merged: array-like of shape (n_j, n_features)
            merged Cluster
    """

    dim = (len(cluster_i[0]))
    half_n_i = math.floor(len(cluster_i) / 2)
    half_n_j = math.floor(len(cluster_j) / 2)
    n_merged = min(half_n_i, half_n_j)

    # Project points onto specified line

    c_i_mean_projected = np.matmul(cluster_i.mean(axis=0), direction)
    c_j_mean_projected = np.matmul(cluster_j.mean(axis=0), direction)

    cluster_i_centered = cluster_i - cluster_i.mean(axis=0)
    cluster_j_centered = cluster_j - cluster_j.mean(axis=0)

    c_i_projected = np.matmul(cluster_i_centered, direction) + c_i_mean_projected
    c_j_projected = np.matmul(cluster_j_centered, direction) + + c_j_mean_projected

    # concatenate points with the distances of their projection to the mean from the other cluster and sort them
    dist_i = np.array([np.absolute(c_i_projected - c_j_mean_projected)]).T
    c_i_distances = np.concatenate((cluster_i, dist_i), axis=1)
    dist_j = np.array([np.absolute(c_j_projected - c_i_mean_projected)]).T
    c_j_distances = np.concatenate((cluster_j, dist_j), axis=1)

    c_i_distances = c_i_distances[c_i_distances[:, dim].argsort()]
    c_j_distances = c_j_distances[c_j_distances[:, dim].argsort()]

    if n_merged == half_n_i:
        c_i_distances = c_i_distances[:n_merged, :dim]
        j_indices = random.sample(range(half_n_j), k=n_merged)
        c_j_distances = c_j_distances[j_indices, :dim]

    else:
        c_j_distances = c_j_distances[:n_merged, :dim]
        i_indices = random.sample(range(half_n_i), k=n_merged)
        c_i_distances = c_i_distances[i_indices, :dim]

    c_merged = np.concatenate((c_i_distances, c_j_distances), axis=0)

    return c_merged
        
def _compare_variances(var_i, var_var_i,  var_j, var_var_j, var_m, safety_margin):
    """
    This function compares the variances of three clusters (with the two-sample welch test)
    :param var_i: float
            variance of cluster i
    :param var_var_i: float
            variance of variance of cluster i
    
    :param var_j: float
            variance of cluster j
    :param var_var_j: float
            variance of variance of cluster j
    
    :param var_m: float
            variance of cluster merged
    
    :param safety_margin: float
            multiple of standart deviation as margin of saftey
    :return: boolean
            Test-decision (True = true clusters, False = false clusters)
    """



 
    std_i=np.sqrt(var_var_i)
    std_j=np.sqrt(var_var_j)
    if (var_i + safety_margin * std_i < var_m) and (var_j + safety_margin * std_j < var_m):
        return True
    else:
        return False
