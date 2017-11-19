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

import pickle
import time

import numpy as np

from rbm.partition_estimator import PartitionEstimator
from rbm.rbm import MCLVKOptimizer, RBM
from util.config import MODEL_FOLDER
from util.log import Log
from util.utils import Util


class RBMUtil:
    @staticmethod
    def run_tours(rbm: RBM, data):
        Log.info("Logging tours for the trained model")
        optimizer = MCLVKOptimizer(rbm, data, rbm.batch_size, -1, 1)
        optimizer.sample_tour_length_distribution()

    @staticmethod
    def save_model(rbm: RBM, filename):
        filename = 'model_%s_%s' % (filename, time.time())
        filename = MODEL_FOLDER + "%s.model" % filename
        with open(filename, 'wb') as fp:
            pickle.dump(rbm, fp)
            Log.info("Model dumped to %s" % filename)
        return filename

    @staticmethod
    def compute_likelihood(rbm: RBM, data, test, num_hidden, width, height):
        Log.info("init")
        pe = PartitionEstimator(rbm.W, num_hidden, width * height)
        data = Util.add_bias_coefficient(data)
        test = Util.add_bias_coefficient(test)
        Log.info("init")
        free_energy = pe.marginal_cuda(data, pe.NT.visible, pe.RT.log_mean)
        free_energy_t = pe.marginal_cuda(test, pe.NT.visible, pe.RT.log_mean)
        Log.info("Free Energy")

        Z = pe.partition_cuda()  # Get the actual partition function
        Log.info("Z")
        L = free_energy - np.log(Z)  # Average Log Likelihood
        Lt = free_energy_t - np.log(Z)

        Log.var(Z=Z, L=L, Lt=Lt)
        return Z, L, Lt

    @staticmethod
    def compute_compare_z(data,
                          model_file="/homes/mkakodka/ztask/model_15epochonly_1505109449.071542.model",
                          Z=1.4637000889549454e+100,
                          output="/homes/mkakodka/ztask/model_15epochonly_1505109449.071542.op"):
        with open(model_file, 'rb') as fp:
            rbm = pickle.load(fp)
            optimizer = MCLVKOptimizer(rbm, data, rbm.batch_size, 99, 1)
            z = optimizer.compute_Z(Z)
            with open(output, 'wb') as wfp:
                pickle.dump(z, wfp)
