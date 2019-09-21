'''
K-means clustering and subset selection functions
(actual data mining tools)
'''
from collections import defaultdict
from math import ceil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

	# prune all but the three columns with smallest entropy value, return the pruned dataset
	sorted_by_entropy = sorted(entropy_dict.items(), key=lambda x: x[1])[:3]
	columns_with_min_entropy = [val[0] for val in sorted_by_entropy]
	data_after_subspace_selection = data[:,columns_with_min_entropy]

	return data_after_subspace_selection


def get_euclidean_distance(p1, p2):
	sum_of_squares = sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))])
	dist = np.sqrt(sum_of_squares)
	return dist


def get_initial_centroid_indices(data, k):
	centroid_indices = np.random.choice(range(len(data)), k, replace=False)
	return centroid_indices


def assign_objects_to_clusters(data, centroid_indices):
	assignment_dict = defaultdict(list)
	non_centroid_indices = set(range(len(data))) - set(centroid_indices)
	for object_index in non_centroid_indices:
		distance_dict = {}
		for centroid_index in centroid_indices:
			distance_dict[centroid_index] = get_euclidean_distance(data[centroid_index], data[object_index])
		winning_centroid_index, _ = sorted(distance_dict.items(), key=lambda x: x[1])[0]
		assignment_dict[winning_centroid_index].append(object_index)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	m = {0: 'Reds', 1: 'Blues', 2: 'Greens'}
	for i, (ci, il) in enumerate(assignment_dict.items()):
		xdata = data[il, 0]
		ydata = data[il, 1]
		zdata = data[il, 2]
		ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap=m[i])
		ax.scatter3D([data[ci][0]], [data[ci][1]], [data[ci][2]], c=[data[ci][2]], cmap='Accent')
	plt.savefig('file.png')

