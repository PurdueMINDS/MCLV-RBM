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

import json

import numpy as np
import torch
from scipy import sparse


class Util:
    @staticmethod
    def dictize(**kwargs):
        return kwargs

    @staticmethod
    def scale_to_unit_interval(ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_cuda(x):
        return 1.0 / (1 + torch.exp(-x))

    @staticmethod
    def safe_divide(num, den):
        if den == 0:
            return den
        else:
            return num / den

    @staticmethod
    def ccdf(x):
        x = np.array(x)
        x_values = sorted(list(set(x)))
        y = np.array([float(len(x[x >= val])) for val in x_values])
        y = Util.safe_divide(y, sum(y))
        return x_values, y

    @staticmethod
    def empty(obj):
        return obj is None or len(obj) == 0

    @staticmethod
    def isnone(obj, def_obj):
        return def_obj if obj is None else obj

    @staticmethod
    def add_bias_coefficient(an_array):
        if isinstance(an_array, sparse.csr_matrix):
            bias = sparse.csr_matrix(np.ones((an_array.shape[0], 1)))
            csr = sparse.hstack([bias, an_array]).tocsr()
        else:
            csr = np.insert(an_array, 0, 1, 1)
        return csr

    @staticmethod
    def shuffle(an_array):
        np.random.shuffle(an_array)
        return an_array

    @staticmethod
    def chunks(arr, step):
        arr = list(arr)
        l = len(arr)
        return [arr[i:min(i + step, l)] for i in range(0, l, step)]

    @classmethod
    def dict_to_json(cls, tour_lengths):
        return json.dumps(tour_lengths)

    @classmethod
    def json_to_dict(cls, tour_lengths):
        return json.loads(tour_lengths)

    @classmethod
    def put_or_add(cls, a_dict, key, value):
        if key not in a_dict:
            a_dict[key] = value
        else:
            a_dict[key] += value

    @classmethod
    def is_different(cls, h1, h2):
        return not (h1 == h2).all()


if __name__ == '__main__':
    # print(Util.ccdf([1,2,2,2, 3]))

    x = np.array([1, 2, 3, 4, 5])
    print(Util.sigmoid(x))
    x = torch.from_numpy(x).double()
    print(Util.sigmoid_cuda(x))
