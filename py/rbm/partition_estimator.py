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


import itertools
from enum import Enum

import numpy as np
import torch

from util.config import GPU_LIMIT, USE_GPU, PIN, GPU_ID
from util.log import Log


class PartitionEstimator:
    class NT(Enum):
        hidden = 1
        visible = 2

    class RT(Enum):
        array = 1
        sum = 2
        log_mean = 3

    def __init__(self, W, num_hidden, num_visible):
        self.W, self.num_hidden, self.num_visible = W, num_hidden, num_visible

    def partition_cuda(self):
        nt = self.NT.hidden if self.num_hidden < self.num_visible else self.NT.visible
        num = self.num_hidden if nt else self.num_visible
        if num < GPU_LIMIT:
            space = np.array(list(itertools.product([0, 1], repeat=num)))
            space = np.insert(space, 0, 1, 1)
            return self.marginal_cuda(space, nt, self.RT.sum)
        else:
            space1 = np.array(list(itertools.product([0, 1], repeat=num - GPU_LIMIT)))
            space2 = np.array(list(itertools.product([0, 1], repeat=GPU_LIMIT)))
            out = 0.0
            for i in range(space1.shape[0]):
                can_space = np.repeat(np.array([space1[i, :]]), space2.shape[0], axis=0)
                space = np.concatenate((can_space, space2), axis=1)
                space = np.insert(space, 0, 1, 1)
                _out = self.marginal_cuda(space, nt, self.RT.sum)
                out += _out
                if i % 100 == 0:
                    Log.var(done=i, of=space1.shape[0])
            return out

    def marginal_cuda(self, x, nt: NT, rt: RT):  # Training RBMs Eq(20)
        # h samples x dimension
        w = torch.from_numpy(np.array(self.W, dtype=np.float64))
        x = torch.from_numpy(np.array(x, dtype=np.float64))
        if USE_GPU:
            if PIN:
                w = w.pin_memory()
                x = x.pin_memory()
            w = w.cuda(device=GPU_ID, async=PIN)
            x = x.cuda(device=GPU_ID, async=PIN)
        if nt == self.NT.hidden:
            wx = torch.matmul(x, w.transpose(0, 1))
        else:
            wx = torch.matmul(x, w)

        # wx samples x alt_dimension
        ones = torch.ones(wx.shape[0], wx.shape[1])
        ones[:, 0] = 0

        if USE_GPU:
            if PIN:
                ones = ones.pin_memory()
            ones = ones.cuda(device=GPU_ID, async=PIN)
        ones = ones.double() + torch.exp(wx).double()

        marginal_probabilities = ones.prod(dim=1)
        if rt == self.RT.array:
            return marginal_probabilities.cpu().numpy()
        elif rt == self.RT.sum:
            return marginal_probabilities.sum()
        elif rt == self.RT.log_mean:
            return torch.log(marginal_probabilities).mean()
