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

from __future__ import print_function

import pickle

from rbm.rbm import *
from rbm.rbmutil import RBMUtil
from util.config import *

np.set_printoptions(threshold=0, suppress=True)


class Main:
    @classmethod
    def main(cls):
        data = np.load(MNIST_FOLDER + "data.npy")
        test = np.load(MNIST_FOLDER + "test.npy")

        num_hidden = int(ARGS.num_hidden)
        Log.var(data_length=len(data))
        data = data[:10000] if LOCAL else data

        phase = Phase[ARGS.PHASE]
        if phase == Phase.Z_COMPUTATION:
            RBMUtil.compute_compare_z(data)
            return
        elif phase == Phase.COMPUTE_LIKELIHOOD or phase == Phase.RUN_TOURS:
            with open(MODEL_FOLDER + ARGS.filename, 'rb') as filename:
                rbm = pickle.load(filename)
                assert isinstance(rbm, RBM)
                if phase == Phase.COMPUTE_LIKELIHOOD:
                    RBMUtil.compute_likelihood(rbm.W, data, test, num_hidden, WIDTH, HEIGHT)
                elif phase == Phase.RUN_TOURS:
                    RBMUtil.run_tours(rbm, data)
            return
        elif phase == Phase.DATA:
            iteration_input = int(ARGS.iteration)
            if iteration_input >= 0:
                iterations = [iteration_input]
            else:
                iterations = range(int(ARGS.RUNS))

            for iteration in iterations:
                fitted = cls.get_rbm().fit(data, test, iteration)

                if LOG_TOUR:
                    RBMUtil.run_tours(fitted, data)

                filename = RBMUtil.save_model(fitted, ARGS.filename)

                if ARGS.final_likelihood:
                    Z, L, Lt = RBMUtil.compute_likelihood(fitted, data, test, num_hidden, WIDTH, HEIGHT)
                    SQLite().insert_final_log(model_location=filename, Z=Z, L_train=L, L_test=Lt, name=ARGS.filename,
                                              fitted=fitted, iteration=iteration, iterations=len(iterations))

    @classmethod
    def get_rbm(cls):
        return RBM(
            WIDTH * HEIGHT,
            int(ARGS.num_hidden),
            warmup_epochs=int(ARGS.warmup),
            cdk=int(ARGS.cdk),
            mclvk=int(ARGS.mclvk),
            weight_decay=float(ARGS.weight_decay),
            momentum=float(ARGS.momentum),
            learning_rate=float(ARGS.learning_rate),
            plateau=int(ARGS.plateau),
            max_epochs=int(ARGS.total_epochs),
            batch_size=int(ARGS.mini_batch_size),
            method=Method[ARGS.method],
            name=str(ARGS),
            supernode_samples=int(ARGS.supernode_samples)
        )


if __name__ == "__main__":
    print(ARGS)
    Main().main()
