'''
K-means clustering and subset selection functions
(actual data mining tools)
'''
from collections import defaultdict
from math import ceil
import numpy as np


def prune_dimensions(data):
	'''
	Use subspace selection to prune all but the three columns
	with the minimum entropy values
	'''
	entropy_dict = {}
	column_length = data.shape[1]

	# evaluate dataset, column by column
	for index, column in enumerate(data.T):

		# delta = maximum value in column - minimum value in column / 10
		min_column = min(column)
		delta = (max(column) - min_column) / 10

		# put values into the 10 buckets (respective deltas)
		val_bucket_dict = defaultdict(list)
		for val in column:
			if val == min_column:
				bucket = 1
			else:
				bucket = ceil((val - min_column) / delta)
			val_bucket_dict[bucket].append(val)

		# calculate density in each bucket then update total entropy over the column
		entropy = 0		
		for vals in val_bucket_dict.values():
			density = len(vals) / column_length
			entropy -= density * np.log(density)
		entropy_dict[index] = entropy
	print('entropy dict {}'.format(entropy_dict))
	# prune all but the three columns with smallest entropy value, return the pruned dataset
	sorted_by_entropy = sorted(entropy_dict.items(), key=lambda x: x[1])[:3]
	columns_with_min_entropy = [val[0] for val in sorted_by_entropy]
	data_after_subspace_selection = data[:,columns_with_min_entropy]

	return data_after_subspace_selection


def get_euclidean_distance(p1, p2):
	sum_of_squares = sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))])
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

