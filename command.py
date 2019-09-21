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
		data = cluster.prune_dimensions(data)
		print(data)


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
		print('running clustering algorithm on {}...'.format(data_file))
		data, params = utils.prepare_data(data_file)
		handle(data, *params)
