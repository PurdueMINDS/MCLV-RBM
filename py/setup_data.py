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

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from util.config import MNIST_FOLDER

np.set_printoptions(threshold=0, suppress=True)

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data, _, test, _ = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    np.save(MNIST_FOLDER + "data.npy", data)
    np.save(MNIST_FOLDER + "test.npy", test)
