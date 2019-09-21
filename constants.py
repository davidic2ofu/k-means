'''
Various constants used by the program
'''
import os


DEFAULT_DATA_FILE_NAME = 'test.dat'
PROGRAM_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE_PATH = PROGRAM_DIRECTORY + os.sep + DEFAULT_DATA_FILE_NAME

MIN_K = 3
MAX_K = 10
MIN_OBJECTS = 50
MAX_OBJECTS = 500
MIN_DIMENSIONS = 2
MAX_DIMENSIONS = 9
