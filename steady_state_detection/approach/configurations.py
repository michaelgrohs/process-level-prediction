# This file contains all model (hyper)parameters and configuration
from pathlib import Path
import os
###################################################################
# DEFAULT DIRECTORIES
###################################################################
# A general input folder for the event logs in xes-format for the main script
DEFAULT_INPUT_DIR = os.path.join("input")

# A default output folder for any approach execution
DEFAULT_INTERIM_DIR = os.path.join("interim")

# A default output folder for any approach execution
#DEFAULT_OUTPUT_DIR = os.path.join("output")
DEFAULT_OUTPUT_DIR = os.path.join("output")

# A default output folder for any approach execution
DEFAULT_PARAMETER_DIR = os.path.join("approach/ssd_methods/offline_methods/ssd_methods_parameters.yml")


###################################################################
# FRAMEWORK configurations
###################################################################
WINDOW_SIZE = ['W']
SSD_APPROACH = [{
    'approach': 'apply_ssd_using_rolling_windows',
    'arguments': {'long_interval': [25],
                  'short_interval': [5],
                  'std_threshold': [0.5],
                  'detect': ['both']}
}]

SSD_AGGREGATION = 'kernel_4'
KERNELS = [4]
KERNEL_CONSENS = 0.7
SS_TRACE_THRESHOLD = 0.8

###################################################################
# Preprocessing configurations
###################################################################
ATTR_CASE = 'case:concept:name'
ATTR_ACTIVITY = 'concept:name'
ATTR_TIME = 'time:timestamp'
ATTR_LIFECYCLE = 'lifecycle:transition'
ATTR_RESOURCE = 'org:resource'
ATTR_ORG = 'org:org'
ATTR_WINDOW = 'window'
ATTRIBUTES = [ATTR_CASE, ATTR_ACTIVITY, ATTR_TIME, ATTR_RESOURCE, ATTR_ORG]


###################################################################
# Multiprocessing
###################################################################
NUM_CORES = 170
SHOW_PLOTS = False
SAVE_PLOTS = False
TESTING = False
EVALUATION = False