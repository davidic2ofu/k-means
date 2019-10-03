'''
K-means clustering and subspace selection functions
(actual data mining tools)
'''
from itertools import combinations
import numpy as np


def get_entropy_over_space(space):
	'''
	1. Create buckets that keep up with number of points in each subspace.
	   The subspaces are defined by deltas in each dimension set to 1/10 of the
	   dimension's range.
	   - i.e., if space is 3-d, then there will be 10^3 subspaces (and buckets)
	2. Calculate density of each subspace
	3. Calculate entropy of the whole space based on densities of the subspaces
	'''
	deltas = [(min(column), (max(column) - min(column)) / 10) for column in space.T]
	num_dims = space.shape[1]
	buckets = np.zeros(10 ** num_dims)
	for obj in space:
		bucket_index = 0
		for i, coordinate in enumerate(obj):
			shifted = coordinate - deltas[i][0]
			if deltas[i][1] == 0:
				continue
			index = shifted // deltas[i][1]
			if index == 10:
				index = 9
			index *= 10 ** i
			bucket_index += index
		buckets[int(bucket_index)] += 1
	densities = buckets / buckets.size
	entropy = -np.sum(densities * np.ma.log2(densities).filled(0))
	return entropy


def prune_dimensions_brute_force(data):
	'''
	Brute force approach:
	calculate entropy of each 3-d combination of subspaces in dataset;
	select subspace with lowest entropy value
	'''
	num_columns = data.shape[1]
	all_3d_combinations = combinations(range(num_columns), 3)
	space_entropy_map = {}
	for comb in all_3d_combinations:
		space = data[:,comb]
		space_entropy_map[comb] = get_entropy_over_space(space)
	best_comb = min(space_entropy_map, key=space_entropy_map.get)
	return data[:,best_comb]


def get_euclidean_distance(p1, p2):
	p1, p2 = map(np.array, (p1, p2))
	sum_of_squares = np.sum((p1 - p2) ** 2)
	dist = np.sqrt(sum_of_squares)
	return dist


def get_sum_of_squared_error(cluster_dict):
	sum = 0
	for centroid, cluster in cluster_dict.items():
		for obj in cluster:
			dist = get_euclidean_distance(obj, centroid)
			sum += dist ** 2
	return sum


def assign_initial_centroids(data, k):
	centroid_indices = np.random.choice(range(len(data)), k, replace=False)
	cluster_dict = {}
	for centroid_index in centroid_indices:
		coordinates = tuple((x for x in data[centroid_index]))
		cluster_dict[coordinates] = []
	return cluster_dict


def reassign_centroids(cluster_dict):
	new_cluster_dict = {}
	for i in range(len(cluster_dict)):
		column = list(cluster_dict.values())[i]
		col_mean = np.mean(column, axis=0)
		new_cluster_dict[tuple(col_mean)] = []
	return new_cluster_dict


def assign_points_to_clusters(cluster_dict, data):
	for datum in data:
		distance_dict = {}
		for candidate_centroid in cluster_dict.keys():
			distance_dict[candidate_centroid] = get_euclidean_distance(candidate_centroid, datum)
		winning_centroid, _ = sorted(distance_dict.items(), key=lambda x: x[1])[0]
		cluster_dict[winning_centroid].append(datum)
	return cluster_dict

