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

import time
from collections import defaultdict
from enum import Enum

import numpy as np
import torch

from bean.metric import Metric
from bean.times import Times
from rbm.partition_estimator import PartitionEstimator
from util.config import GPU_LIMIT, USE_GPU, PIN, GPU_ID
from util.log import Log
from util.sqlite import SQLite
from util.utils import Util


class Method(Enum):
    CD = 1
    MCLV = 2
    PCD = 3


class Optimizer:
    def __init__(self, rbm, batch_size, epoch):
        """

        :type rbm: RBM
        """
        self.rbm = rbm
        self.batch_size = batch_size
        self.pe = PartitionEstimator(rbm.W, rbm.num_hidden, rbm.num_visible)
        self.epoch = epoch

        self.total_tours = 0
        self.completed_tours = 0
        self.diverse_tours = 0
        pass

    def optimize(self, v, h, hs, mini_batch_id):
        raise NotImplementedError()

    def update_weights(self, pos_associations, neg_associations):
        # Update weights.
        lr = self.rbm.learning_rate * 1 / np.ceil((1 + self.epoch) / self.rbm.plateau)
        w_update = lr * (
            ((pos_associations - neg_associations) / self.batch_size)
            - (self.rbm.weight_decay * self.rbm.W)
        ) + self.rbm.momentum * self.rbm.last_change

        self.rbm.W += w_update
        self.rbm.last_change = w_update

    def sample_next_state(self, tour_indexes, v, h, hs):
        v[tour_indexes] = self.rbm.visible_from_hidden(hs[tour_indexes])
        h[tour_indexes] = self.rbm.hidden_from_visible(v[tour_indexes])
        hs[tour_indexes] = self.rbm.hard_samples(h[tour_indexes])
        return v, h, hs

    def get_vh_cuda(self, v, h):
        v = torch.from_numpy(v)
        h = torch.from_numpy(h)
        if USE_GPU:
            if PIN:
                v = v.pin_memory()
                h = h.pin_memory()
            v = v.cuda(device=GPU_ID, async=PIN)
            h = h.cuda(device=GPU_ID, async=PIN)
        return torch.matmul(v.transpose(0, 1).double(), h.double()).cpu().numpy()


class CDKOptimizer(Optimizer):
    def __init__(self, rbm, batch_size, epoch, persistent):
        super().__init__(rbm, batch_size, epoch)
        self.persistent = persistent

    def optimize(self, v, h, hs, mini_batch_id):
        times = Times()
        pos_associations = self.get_vh_cuda(v, h)
        if self.persistent and self.rbm.persistent_chains is not None:
            hs = self.rbm.persistent_chains

        for _ in range(self.rbm.cdk):
            v, h, hs = self.sample_next_state(list(range(self.batch_size)), v, h, hs)
        neg_associations = self.get_vh_cuda(v, h)
        if self.persistent:
            self.rbm.persistent_chains = hs

        self.update_weights(pos_associations, neg_associations)
        times.add('CDK')
        return times


class MCLVKOptimizer(Optimizer):
    def __init__(self, rbm, train, batch_size, epoch, iteration, supernode_samples=None):
        super().__init__(rbm, batch_size, epoch)
        self.train = train
        self.iteration = iteration
        self.supernode, self.supernode_matrix, self.supernode_size = self.gen_supernode(supernode_samples)

    def optimize(self, v, h, hs, mini_batch_id):
        times = Times()
        pos_associations = self.get_vh_cuda(v, h)
        times.add('init')

        history, total_tours, completed_tours, diverse_tours = self.run_tours()
        self.total_tours += total_tours
        self.completed_tours += completed_tours
        self.diverse_tours += diverse_tours
        times.add('run_tours')

        self.generate_update(history, pos_associations)
        times.add('gen_update')
        if len(history) == 0:
            self.supernode, self.supernode_matrix, self.supernode_size = self.gen_supernode()
        times.add('gen_supernode')
        return times

    def sample_tour_length_distribution(self):
        Log.info("Generating Tour Length Distribution")
        tour_limit = 1000
        tour_supernode_sizes = [1, 4, 7]
        num_samples = 150

        for supernode_samples in tour_supernode_sizes:
            supernode, supernode_matrix, supernode_size = self.gen_supernode(supernode_samples)
            for mini_batch_id in range(num_samples):
                history, total_tours, completed_tours, diverse_tours = self.run_tours(tour_limit, supernode,
                                                                                      supernode_matrix,
                                                                                      experimental=True)
                tour_length_dist = defaultdict(int)
                for h in history:
                    tour_length = len(h) - 1
                    tour_length_dist[tour_length] += 1

                SQLite().insert_tour_length_data(
                    config=self.rbm.name,
                    iter=self.iteration,
                    hidden=self.rbm.num_hidden,
                    k=self.rbm.mclvk,
                    epoch=self.epoch,
                    mini_batch_id=mini_batch_id,
                    total_tours=total_tours,
                    completed_tours=completed_tours,
                    diverse_tours=diverse_tours,
                    supernode_size=supernode_size,
                    supernode_samples=supernode_samples,
                    tour_lengths=Util.dict_to_json(tour_length_dist)
                )
                Log.info("Sampled supernode_samples=%s mini_batch_id=%s", supernode_samples, mini_batch_id)

    def compute_Z(self, Z):
        Log.info("Generating Tour Length Distribution")
        num_samples = 1000
        self.batch_size = 1

        supernode_samples = 1
        supernode, supernode_matrix, supernode_size = self.gen_supernode(supernode_samples)
        sum_length = 0.0
        count_length = 0.0
        Z_Map = []
        Z_s = self.pe.marginal_cuda(supernode_matrix, self.pe.NT.hidden, self.pe.RT.array).sum()
        Log.var(Z_s=Z_s)
        for mini_batch_id in range(num_samples):
            history, _, _, _ = self.run_tours(100000, supernode,
                                              supernode_matrix, experimental=False)
            tl = [len(h) - 1 for h in history]
            sum_length += sum(tl)
            count_length += len(tl)
            xi = sum_length / count_length
            Z_est = xi * Z_s
            Z_err = abs(Z - Z_est)
            Z_Map.append(Util.dictize(count_length=count_length, Z_est=Z_est, Z_err=Z_err, Z=Z, xi=xi))
            Log.var(count_length=count_length, Z_est=Z_est, Z_err=Z_err, Z=Z, xi=xi)
        return Z_Map

    def run_tours(self, tour_limit=None, supernode=None, supernode_matrix=None, experimental=False):
        tour_limit = Util.isnone(tour_limit, self.rbm.mclvk)
        supernode_matrix = Util.isnone(supernode_matrix, self.supernode_matrix)
        supernode = Util.isnone(supernode, self.supernode)
        num_tours = self.batch_size

        if not experimental:
            probabilities = self.pe.marginal_cuda(supernode_matrix, self.pe.NT.hidden, self.pe.RT.array)
            probabilities = probabilities / probabilities.sum()
            idx = np.random.choice(len(supernode_matrix),
                                   size=num_tours, p=probabilities,
                                   replace=True)
            hs = np.array(supernode_matrix[idx])
        else:
            h = self.rbm.hidden_from_visible(self.train)
            hs = self.rbm.hard_samples(h)
            idx = np.random.choice(len(hs), size=num_tours)
            hs = hs[idx]

        tour_indexes = list(range(num_tours))
        tour_start = hs.copy()  # Set tour start as first sampled hidden states
        v = self.rbm.visible_from_hidden(hs)
        h = self.rbm.hidden_from_visible(v)

        total_tours = len(hs)
        finished_tours = np.zeros(num_tours)
        # The initial stored values of v and h will not be used to compute the gradient
        history = [[State(hs[i], v[i], h[i])] for i in tour_indexes]
        # also stops if all tours have finished
        for ii in range(tour_limit):
            hs[tour_indexes] = self.rbm.hard_samples(h[tour_indexes])
            v[tour_indexes] = self.rbm.visible_from_hidden(hs[tour_indexes])
            h[tour_indexes] = self.rbm.hidden_from_visible(v[tour_indexes])

            # checks which tours have just finished
            just_finished_tours = np.array(
                [np.array_equal(tour_start[k], hs[k]) for k in range(num_tours)])

            for ti in tour_indexes:
                if self.get_state_hash(hs[ti]) in supernode:
                    just_finished_tours[ti] = True
                history[ti].append(State(hs[ti], v[ti], h[ti]))

            finished_tours = np.logical_or(finished_tours, just_finished_tours)
            # removes index from tour_indexes in case the tour has just finished
            tour_indexes = [i for i in tour_indexes if not finished_tours[i]]

            if Util.empty(tour_indexes):
                break
                # Log.info("%s steps done, %s tours still remaining", ii, len(tour_indexes))
        history = [h for f, h in zip(finished_tours, history) if f]
        completed_tours = len(history)
        diverse_tours = sum([1 for h in history if Util.is_different(h[0].hs, h[len(h) - 1].hs)])
        return history, total_tours, completed_tours, diverse_tours

    @staticmethod
    def get_state_hash(row):
        return tuple(row)

    def gen_supernode(self, supernode_samples=None):
        if supernode_samples is None:
            supernode_samples = self.rbm.supernode_samples
        Log.info("Generating supernode")
        start = time.time()

        supernode = dict()
        supernode_sizes = []
        for _ in range(supernode_samples):
            h = self.rbm.hidden_from_visible(self.train)
            hs = self.rbm.hard_samples(h)
            for row in hs:
                supernode[self.get_state_hash(row)] = row

            supernode_sizes.append(len(supernode))

        supernode_matrix = np.array([row for row in supernode.values()])
        supernode_size = len(supernode)

        end = time.time()
        Log.var(supernode_sizes=supernode_sizes, time=end - start)
        return supernode, supernode_matrix, supernode_size

    def generate_update(self, history, pos_associations):
        neg_associations = self.get_negative_associations(history, pos_associations)
        if neg_associations is not None:
            self.update_weights(pos_associations, neg_associations)

    def get_negative_associations(self, history, pos_associations, weight_by_batch_size=True):
        neg_associations = pos_associations.copy() * 0  # Initialize negative associations
        tour_lengths = []
        for hist_element in history:
            # don't use last element
            # so that the final stored values of v and h are not used to compute the gradient
            v_tour = np.matrix([s.v for s in hist_element[:-1]])
            h_tour = np.matrix([s.h for s in hist_element[:-1]])
            hs_tour = np.matrix([s.hs for s in hist_element[:-1]])
            tour_length = v_tour.shape[0]
            neg_associations += self.get_vh_cuda(v_tour, hs_tour)
            tour_lengths.append(tour_length)
        if Util.empty(tour_lengths):
            neg_associations = None
        else:
            sum_tour_length = float(sum(tour_lengths))
            neg_associations *= ((self.batch_size if weight_by_batch_size else 1.0) / sum_tour_length)
        return neg_associations


class State:
    def __init__(self, hs, v, h):
        self.v, self.h, self.hs = v.copy(), h.copy(), hs.copy()


class RBM:
    def __init__(self, num_visible=None, num_hidden=None, W=None, learning_rate=0.1,
                 weight_decay=0.0, cdk=1,
                 momentum=0.0, warmup_epochs=1, mclvk=200, plateau=10000000, max_epochs=1,
                 batch_size=100,
                 method=Method.MCLV, name=None, supernode_samples=1):
        if W is None:
            self.W = np.random.uniform(-.1, 0.1, (num_visible,
                                                  num_hidden)) / np.sqrt(num_visible + num_hidden)
            self.W = np.insert(self.W, 0, 0, axis=1)
            self.W = np.insert(self.W, 0, 0, axis=0)
        else:
            self.W = W

        # Config
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        self.plateau = plateau
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cdk = cdk
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.method = method
        self.name = name

        # MCLV Specific
        self.warmup_epochs = warmup_epochs
        self.mclvk = mclvk
        self.supernode_samples = supernode_samples

        # PCD Specific
        self.persistent_chains = None

        # House Keeping
        self.last_change = 0
        self.metric = Metric()

    def init_b(self, train):
        """
            Initialize the weights based on training data.
        :param train:
        :return:
        """
        eps = 1e-2
        p = train.mean(axis=0)
        p = np.log((p + eps) / (1 - p))
        self.W[1:, 0] = p

    def fit(self, train, test, iteration=None):
        Log.info("                                                                 ")
        Log.info("               Starting fit with opt_type=%s                     ", self.name)
        Log.info("                                                                 ")

        self.init_b(train)

        train = Util.add_bias_coefficient(train)
        test = Util.add_bias_coefficient(test)

        self.evaluate(train, test, epoch_time=-1, epoch_number=-1, tour_detail=None)
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            idx_list = Util.shuffle(np.arange(train.shape[0]))

            if self.method == Method.MCLV \
                    and epoch >= self.warmup_epochs:
                if epoch == self.warmup_epochs:
                    Log.info("Warm up complete, switching to %s" % (self.method))
                optimizer = MCLVKOptimizer(self, train, self.batch_size, epoch, iteration)
            else:
                optimizer = CDKOptimizer(self, self.batch_size, epoch, persistent=self.method == Method.PCD)

            time_sum = defaultdict(float)
            for j in range(int(train.shape[0] / self.batch_size)):
                # Sequential Samples of size batch size :
                idx = idx_list[j * self.batch_size:(j + 1) * self.batch_size]
                v = train[idx]
                h = self.hidden_from_visible(v)
                hs = self.hard_samples(h)
                times = optimizer.optimize(v, h, hs, j)
                for ev, ti in times.compute().items():
                    time_sum[ev] += ti
            Log.var(time_sum=time_sum)

            # Report Epoch Time
            epoch_end = time.time()
            self.evaluate(train, test, epoch_time=(epoch_end - epoch_start), epoch_number=epoch
                          , tour_detail=(optimizer.completed_tours, optimizer.total_tours, optimizer.diverse_tours))
        return self

    @staticmethod
    def hard_samples(p):
        return np.array(np.random.binomial(1, p=p))

    def hidden_from_visible(self, V):
        return self._sample_alternate_state(V, True)

    def visible_from_hidden(self, H):
        return self._sample_alternate_state(H, False)

    def _sample_alternate_state(self, s, from_visible):
        w = self.W if from_visible else self.W.T

        if s.shape[1] != w.shape[0]:
            s = Util.add_bias_coefficient(s)
        else:
            s = s
        w = torch.from_numpy(w)
        s = torch.from_numpy(s)

        if USE_GPU:
            if PIN:
                w = w.pin_memory()
                s = s.pin_memory()
            w = w.cuda(device=GPU_ID, async=PIN)
            s = s.cuda(device=GPU_ID, async=PIN)

        p = Util.sigmoid_cuda(torch.matmul(s.double(), w))
        p[:, 0] = 1.0
        return p.cpu().numpy()

    def evaluate(self, data, test, epoch_time, epoch_number, tour_detail):
        # computes partition function over all states and over visible and test states and prints out
        eval_start = time.time()
        pe = PartitionEstimator(self.W, self.num_hidden, self.num_visible)

        # For verification
        # Zv = list(map(pe.visible_partition, data))  # Numerator of likelihood
        # Zt = list(map(pe.visible_partition, test))  # Numerator of likelihood
        # L = (np.log(Zv) - np.log(Z)).mean()  # Average Log Likelihood
        # Lt = (np.log(Zt) - np.log(Z)).mean()  # Average Log Likelihood

        free_energy = pe.marginal_cuda(data, pe.NT.visible, pe.RT.log_mean)
        free_energy_t = pe.marginal_cuda(test, pe.NT.visible, pe.RT.log_mean)

        if self.num_hidden < GPU_LIMIT:
            Z = pe.partition_cuda()  # Get the actual partition function
            L = free_energy - np.log(Z)  # Average Log Likelihood
            Lt = free_energy_t - np.log(Z)
        else:
            L = Lt = None

        error_t = self.get_reconstruction_errors(test)
        error = self.get_reconstruction_errors(data)

        eval_end = time.time()

        self.metric.test.likelihoods.append(Lt)
        self.metric.test.free_energy.append(free_energy_t)
        self.metric.test.reconstruction_errors.append(error_t)

        self.metric.train.likelihoods.append(L)
        self.metric.train.free_energy.append(free_energy)
        self.metric.train.reconstruction_errors.append(error)

        Log.info("%s \n %s", epoch_number, self.metric.get_table(epoch_time, eval_end - eval_start, tour_detail))

    def get_reconstruction_errors(self, data):
        h = self.hidden_from_visible(data)
        hs = self.hard_samples(h)
        recon = self.visible_from_hidden(hs)
        data = torch.from_numpy(data)
        recon = torch.from_numpy(recon)
        if USE_GPU:
            if PIN:
                data = data.pin_memory()
                recon = recon.pin_memory()
            data = data.cuda(device=GPU_ID, async=PIN)
            recon = recon.cuda(device=GPU_ID, async=PIN)
        return np.sqrt(torch.pow(data.float() - recon.float(), 2).sum(dim=1).mean())
