import socket

import numpy as np
import torch
import os

from u.argument_parser import CustomArgumentParser

ARGS = CustomArgumentParser.parse_top_level_arguments()
LOCAL = ARGS.local
AVAILABLE_GPUS = ARGS.gpu_id
BASE_FOLDER = ARGS.base_folder
USE_DOUBLE = ARGS.double
if USE_DOUBLE:
    NP_DT = np.float64
else:
    NP_DT = np.float32

DISCRETE = not ARGS.no_discrete
DEBUG = ARGS.debug
HOSTNAME = socket.gethostname()

print("BASE_FOLDER=%s" % BASE_FOLDER)
print("LOCAL=%s" % LOCAL)

MNIST_FOLDER = BASE_FOLDER + 'py/MNIST_data/'

DATA_FOLDER = BASE_FOLDER + 'data/'
PLOT_OUTPUT_FOLDER = DATA_FOLDER + 'plots/'
MODEL_FOLDER = DATA_FOLDER + 'model/'
IMG_OUTPUT_FOLDER = DATA_FOLDER + 'img/'
LOG_FOLDER = DATA_FOLDER + 'log/'
SQL_FOLDER = DATA_FOLDER + 'sql/'
for d in [DATA_FOLDER, PLOT_OUTPUT_FOLDER, MODEL_FOLDER, IMG_OUTPUT_FOLDER, LOG_FOLDER, SQL_FOLDER]:
    if not os.path.exists(d):
        os.makedirs(d)

SQLITE_FILE = SQL_FOLDER + ARGS.sqlite + '.db'

GPU_LIMIT = int(ARGS.gpu_limit)
GPU_HARD_LIMIT = int(ARGS.gpu_hard_limit)
BATCH_SIZE_LIMIT = int(ARGS.batch_size_limit)
PIN = False
LOG_TOUR = ARGS.log_tour
TOUR_LENGTHS_TABLE = "TOUR_LENGTHS"
WIDTH = 28
HEIGHT = 28

EPOCH_TRAINING_TABLE = "EPOCH_DETAILS"
NAME_TABLE = "NAME_DETAILS"
FINAL_LIKELIHOODS_TABLE = "LIKELIHOOD_DETAIL"
GRADIENT_DETAIL_TABLE = "GRADIENT_DETAIL_NEW_WS"
PARTITION_ESTIMATION_TABLE = "PARTITION_ESTIMATION"
MAIN_THREAD = 'MainThread'

if ARGS.timetest:
    TIME_TEST = True
    NO_DB = True
else:
    TIME_TEST = False
    NO_DB = False

PARALLELISM = ARGS.parallelism


def SET_PARALLELISM():
    if PARALLELISM is not None:
        current = torch.get_num_threads()
        new = int(PARALLELISM)
        torch.set_num_threads(new)
        new = torch.get_num_threads()
        print("Parallelism changed from %s to %s" % (current, new))
    else:
        current = torch.get_num_threads()
        print("Parallelism is %s" % current)


SET_PARALLELISM()
