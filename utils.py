'''
File handling: input and output
'''
import numpy as np

from constants import *


def prepare_data(data_file):
	'''
	Parse the input file
	'''
	with open(data_file, 'r+') as f:
		params = f.readline()
	params = params.split()
	data = np.loadtxt(data_file, skiprows=1)

	return data, params


def build_data_file():
	'''
	Generate data file according to assignments specifications
	to use as input for clustering algorithm
	'''
	k_vals = np.random.choice(range(MIN_K, MAX_K), 2, replace=False)
	k_min = min(k_vals)
	k_max = max(k_vals)

	num_objects = np.random.randint(MIN_OBJECTS, MAX_OBJECTS)
	num_dimensions = np.random.randint(MIN_DIMENSIONS, MAX_DIMENSIONS)

	header = [num_objects, num_dimensions, k_min, k_max]
	objects = np.random.rand(num_objects, num_dimensions) * 100

	with open(DEFAULT_DATA_FILE_PATH, 'w+') as f:
		f.write(' '.join(map(str, header)) + '\n')
		np.savetxt(f, objects)

	msg = 'created data file: ' + DEFAULT_DATA_FILE_PATH

	return msg
