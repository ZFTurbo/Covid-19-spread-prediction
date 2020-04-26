# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os

# How many days to use in lag features
LIMIT_DAYS = 10

# Constants control which transform of target to use
USE_LOG = 1
USE_DIFF = 1
USE_DIV = 1

# GPU usage for training
USE_GPU = 1

# Number of iterations for each GBM. The bigger the better, but more required time
REQUIRED_ITERATIONS = 5

# Days to predict in future
DAYS_TO_PREDICT = 7

# Use this if you need to check real validation (train on data from 7 days ago and test on hold out but already known data)
STEP_BACK = None
# STEP_BACK = 7

# Set to true to use last day CSV for countires (Warning: depends on day time it can still be only partially updated!)
USE_LATEST_DATA_COUNTRY = False
USE_LATEST_DATA_RUS = True

# Some feature control variables
USE_YANDEX_MOBILITY_DATA = True
USE_SIMPLE_LINEAR_FEATURES = True
USE_INTERPOLATION_FEATURES = True
USE_WEEKDAY_FEATURES = True

# All path variables
ROOT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
FEATURES_PATH = ROOT_PATH + 'features/'
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
HISTORY_FOLDER_PATH = MODELS_PATH + "history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
SUBM_PATH = ROOT_PATH + 'subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)