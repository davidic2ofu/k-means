'''
File handling: input and output
'''
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

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
	objects = np.random.rand(num_objects, num_dimensions)

	with open(DEFAULT_DATA_FILE_PATH, 'w+') as f:
		f.write(' '.join(map(str, header)) + '\n')
		np.savetxt(f, objects)

	msg = 'created data file: ' + DEFAULT_DATA_FILE_PATH
	return msg


def generate_output_file(output_list):
	'''
	Generate output file:
	k and sse on each line
	'''
	outfile = PROGRAM_DIRECTORY + os.sep + 'test.res'
	with open(outfile, 'w+') as f:
		for k, sse in output_list:
			f.write('{} {}\n'.format(k, sse))

	msg = 'Generated output file {}'.format(outfile)
	return msg


def delete_existing_scatterplot_figures():
	wildcard = PROGRAM_DIRECTORY + os.sep + '*.png'
	filelist = glob.glob(wildcard)
	for filepath in filelist:
		os.remove(filepath)
	return wildcard if filelist else ''


def display_visuals():
	osx_command = 'open {}'.format(DEFAULT_FIGURE_PATH_WILDCARD)
	os.system(osx_command)


def visualize(cluster_dict, num_dimensions):
	'''
	Generate 2/3D scatterplot figures
	'''
	fig = plt.figure()

	if num_dimensions > 2:
		ax = fig.add_subplot(111, projection='3d')
		for i, (_, point_list) in enumerate(cluster_dict.items()):
			for point in point_list:
				xdata = point[0]
				ydata = point[1]
				zdata = point[2]
				ax.scatter3D(xdata, ydata, zdata, c=MATPLOTLIB_COLORS[i])
	else:
		ax = fig.add_subplot(111)
		for i, (_, point_list) in enumerate(cluster_dict.items()):
			for point in point_list:
				xdata = point[0]
				ydata = point[1]
				ax.scatter(xdata, ydata, c=MATPLOTLIB_COLORS[i])

	filepath = DEFAULT_FIGURE_FILE_PATH.format(len(cluster_dict))
	plt.savefig(filepath)
	
	return filepath
