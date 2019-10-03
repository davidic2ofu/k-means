import argparse
import numpy as np
import os
import sys

import cluster
from constants import *
import utils


parser = argparse.ArgumentParser()

parser.add_argument(
	'-b', '--build_data_file',
	action='store_true',
	help='creates "{}" data file in program directory'.format(DEFAULT_DATA_FILE_NAME),
)

parser.add_argument(
	'-r', '--run',
	action='store_true',
	help='run the clustering algorithm on test data',
)

parser.add_argument(
	'-d', '--data_file',
	action='store',
	default=DEFAULT_DATA_FILE_PATH,
	help='specify path to data file (default is "{}" in program directory)'.format(DEFAULT_DATA_FILE_NAME),
)


def handle(data, num_objects, num_dimensions, k_min, k_max):
	'''
	handle the flow of the algorithm
	'''
	if num_dimensions > 3:
		print('Pruning dataset from {} dimensions to 3...'.format(num_dimensions))
		data = cluster.prune_dimensions_brute_force(data)

	print('Beginning k-means clustering on set of {} objects...'.format(num_objects))

	output_list = []

	for k in range(k_min, k_max + 1):
		
		# prepare initial clusters
		cluster_dict = cluster.assign_initial_centroids(data, k)
		cluster_dict = cluster.assign_points_to_clusters(cluster_dict, data)
		new_sse = cluster.get_sum_of_squared_error(cluster_dict)

		# max 20 iterations to converge
		for _ in range(20):
			sse = new_sse
			cluster_dict = cluster.reassign_centroids(cluster_dict)
			cluster_dict = cluster.assign_points_to_clusters(cluster_dict, data)
			new_sse = cluster.get_sum_of_squared_error(cluster_dict)
			if new_sse == sse:
				break

		print('SSE for k = {}: {}'.format(k, sse))
		output_list.append((k, sse))

		try:
			filepath = utils.visualize(cluster_dict, num_dimensions)
			print('Saved figure {}'.format(filepath))
		except:
			pass

	msg = utils.generate_output_file(output_list)
	print(msg)


if __name__ == '__main__':
	'''
	parse command arguments and direct the program
	'''
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(0)

	args = parser.parse_args()
	arg_dict = vars(args)

	if arg_dict['build_data_file']:
		msg = utils.build_data_file()
		print(msg)
		sys.exit(0)

	if arg_dict['run']:
		data_file = arg_dict['data_file']
		if not os.path.exists(data_file):
			print('"{}" not found on disk.  Create new test data "{}" in program directory with option -b'.format(data_file, DEFAULT_DATA_FILE_NAME))
			sys.exit(0)

		filepaths = utils.delete_existing_scatterplot_figures()
		if filepaths:
			delete_msg = 'Removed {} from the file system'.format(filepaths)
			print(delete_msg)

		print('Preparing {}...'.format(data_file))
		data, params = utils.prepare_data(data_file)

		handle(data, *map(int, params))

		try:
			utils.display_visuals()
		except:
			pass
