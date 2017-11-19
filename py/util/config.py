# Copyright 2017 Bruno Ribeiro, Mayank Kakodkar, Pedro Savarese
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch

from bean.phase import Phase


def parse_top_level_arguments():
    parser = argparse.ArgumentParser(description='Fit RBM to MNIST using different gradient estimators')
    parser.add_argument('--local', '-l', dest='LOCAL', action='store_const',
                        const=True, default=False,
                        help='Enables Local run')
    parser.add_argument('--basefolder', '-b', dest='BASE_FOLDER', action='store'
                        , default='/Users/mkakodka/Code/Research/RBM_V1/',
                        help='Base Folder for all directory paths')
    parser.add_argument('--phase', '-p', dest='PHASE', action='store'
                        , default='DATA',
                        help=str(Phase.__dict__))
    parser.add_argument('-n', dest='RUNS', action='store'
                        , default='1',
                        help='Number of runs')
    parser.add_argument('-iteration', dest='iteration', action='store'
                        , default='-1',
                        help='iteration')
    parser.add_argument('--method', '-m', dest='method', action='store',
                        default="MCLV",
                        help='Method to use')
    parser.add_argument('-sfs', dest='sample_from_supernode', action='store_const',
                        const=True, default=False,
                        help='Sample from supernode for tour distribution')
    parser.add_argument('-cdk', dest='cdk', action='store',
                        default=1,
                        help='contrastive divergence steps limit')
    parser.add_argument('-mclvk', dest='mclvk', action='store',
                        default=1,
                        help='tour length limit')
    parser.add_argument('-wm', dest='warmup', action='store',
                        default=2,
                        help='warmup epochs')
    parser.add_argument('-tot', '--total-epochs', dest='total_epochs', action='store',
                        default=100,
                        help='total epochs')
    parser.add_argument('-mbs', '--mini-batch-size', dest='mini_batch_size', action='store',
                        default=128,
                        help='mini batch size')
    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', action='store',
                        default=0.1,
                        help='learning rate')
    parser.add_argument('--weight-decay', '-wd', dest='weight_decay', action='store',
                        default=0.0,
                        help='weight decay')
    parser.add_argument('--momentum', '-mm', dest='momentum', action='store',
                        default=0.0,
                        help='momentum')
    parser.add_argument('--plateau', '-pt', dest='plateau', action='store',
                        default=1000,
                        help='Robbins Munro Schedule plateau length')
    parser.add_argument('--hidden', dest='num_hidden', action='store',
                        default=16,
                        help='Number of hidden units')
    parser.add_argument('--supernode-samples', '-ss', dest='supernode_samples', action='store',
                        default=1,
                        help='Number of samples to include in the supernode')

    parser.add_argument('--gpu-id', dest='gpu_id', action='store',
                        default=-1,
                        help='gpu_id')
    parser.add_argument('--gpu-limit', dest='gpu_limit', action='store',
                        default=18,
                        help='gpu_limit')
    parser.add_argument('--filename', dest='filename', action='store',
                        default='temp_local',
                        help='filename')

    parser.add_argument('--final-likelihood', dest='final_likelihood', action='store_const',
                        const=True, default=False,
                        help='compute final likelihood')

    parser.add_argument('--log-tour', dest='LOG_TOUR', action='store_const',
                        const=True, default=False,
                        help='LOG_TOUR')

    parser.add_argument('--name', dest='name', action='store',
                        default=None,
                        help='Name this run')
    args = parser.parse_args()
    return args.LOCAL, args.BASE_FOLDER, args


LOCAL, BASE_FOLDER, ARGS = parse_top_level_arguments()

print("Config.BASE_FOLDER=%s" % BASE_FOLDER)
print("Config.LOCAL=%s" % LOCAL)

DATA_FOLDER = BASE_FOLDER + 'data/'
MODEL_FOLDER = BASE_FOLDER + 'data/model/'
OUTPUT_FOLDER = BASE_FOLDER + 'output/'
MNIST_FOLDER = BASE_FOLDER + 'py/MNIST_data/'

PLOT_OUTPUT_FOLDER = BASE_FOLDER + 'plots/'
SQLITE_FILE = DATA_FOLDER + 'results.db'
SERVER_SQLITE_FILE = DATA_FOLDER + 'results_server.db' if LOCAL else SQLITE_FILE

GPU_LIMIT = int(ARGS.gpu_limit)
USE_GPU = torch.cuda.is_available() and not LOCAL
LOG_TOUR = ARGS.LOG_TOUR
TOUR_LENGTHS_TABLE = "TOUR_LENGTH_DISTRIBUTIONS"

# These are hardcoded for the MNIST dataset
WIDTH = 28
HEIGHT = 28

# These options do not work right now, we'll fix them soon
PIN = False
GPU_ID = int(ARGS.gpu_id) if int(ARGS.gpu_id) >= 0 else None
