import time
from collections import defaultdict

import numpy as np
import torch

from b.state import State
from o.optimizer import Optimizer
from u.config import DISCRETE, TOUR_LENGTHS_TABLE
from u.log import Log
from u.sqlite import SQLite
from u.utils import Util


class MCLVKOptimizer(Optimizer):
    def __init__(self, rbm, train, batch_size, epoch, iteration, mclv=True):
        super().__init__(rbm, batch_size, epoch)
        self.train = train
        self.iteration = iteration
        if mclv:
            self.supernode = self.gen_supernode()

    def get_negative_associations(self, v, h, hs, pos_associations, mini_batch_id, times):
        times.add('init')

        history, total_tours, completed_tours, diverse_tours, tour_ccdf = self.run_tours()
        self.total_tours += total_tours
        self.completed_tours += completed_tours
        self.diverse_tours += diverse_tours
        self.tour_ccdf = Util.sum_ccdfs(self.tour_ccdf, tour_ccdf)
        times.add('run_tours')

        neg_associations, _ = self.compute_negative_associations(history)
        times.add('gen_update')
        if len(history) == 0:
            self.supernode = self.gen_supernode()
        times.add('gen_supernode')
        return neg_associations, times

    def run_tours(self, tour_limit=None, supernode=None, sample_uniform=False):
        tour_limit = Util.isnone(tour_limit, self.rbm.mclvk)
        supernode = Util.isnone(supernode, self.supernode)
        num_tours = self.batch_size

        if not sample_uniform:
            probabilities, _ = self.pe.marginal_cuda(supernode.actual, self.pe.NT.hidden, self.pe.RT.normalized)
            idx = np.random.choice(range(supernode.actual.shape[0]),
                                   size=num_tours, p=probabilities.cpu().numpy(),
                                   replace=True)
            hs = supernode.actual[self.rbm.gpu.from_numpy(idx).long()]
        else:
            h = self.rbm.hidden_from_visible(self.train)
            hs = self.rbm.hard_samples(h)
            idx = np.random.choice(len(hs), size=num_tours)
            hs = hs[self.rbm.gpu.from_numpy(idx).long()]

        v = self.rbm.visible_from_hidden(hs)
        if DISCRETE:
            v = self.rbm.hard_samples(v)
        h = self.rbm.hidden_from_visible(v)

        total_tours = len(hs)
        tour_indexes = self.rbm.gpu.long_range(0, total_tours)
        finished_tours = self.rbm.gpu.zeros(total_tours)
        # The initial stored values of v and h will not be used to compute the gradient
        history = [[State(hs[i], v[i], h[i])] for i in tour_indexes]
        # also stops if all tours have finished
        tour_ccdf = [len(tour_indexes)]
        for ii in range(tour_limit):
            hs[tour_indexes] = self.rbm.hard_samples(h[tour_indexes])
            v[tour_indexes] = self.rbm.visible_from_hidden(hs[tour_indexes])
            if DISCRETE:
                v[tour_indexes] = self.rbm.hard_samples(v[tour_indexes])
            h[tour_indexes] = self.rbm.hidden_from_visible(v[tour_indexes])

            finished_tours[tour_indexes] += self.is_in_supernode(hs[tour_indexes], supernode)
            for ti in tour_indexes:
                history[ti].append(State(hs[ti], v[ti], h[ti]))

            # removes index from tour_indexes in case the tour has just finished
            tour_indexes = self.rbm.gpu.long_range(0, total_tours)[finished_tours == 0]
            tour_ccdf.append(len(tour_indexes))
            if Util.empty(tour_indexes):
                break
        history = [h for f, h in zip(finished_tours, history) if f]
        completed_tours = len(history)
        diverse_tours = sum([1 for h in history if Util.is_different(h[0].hs, h[len(h) - 1].hs)])
        return history, total_tours, completed_tours, diverse_tours, tour_ccdf

    def compute_negative_associations(self, history):
        # Initialize negative associations
        neg_associations = self.rbm.gpu.zeros(self.rbm.num_visible + 1, self.rbm.num_hidden + 1)
        tour_lengths = []
        for hist_element in history:
            # don't use last element
            # so that the final stored values of v and h are not used to compute the gradient
            v_tour = torch.cat([s.v[None, :] for s in hist_element[:-1]], 0)
            hs_tour = torch.cat([s.hs[None, :] for s in hist_element[:-1]], 0)
            tour_length = v_tour.shape[0]
            neg_associations += self._get_vh_cuda(v_tour, hs_tour)
            tour_lengths.append(tour_length)
        if Util.empty(tour_lengths):
            neg_associations = None
        else:
            sum_tour_length = float(sum(tour_lengths))
            neg_associations *= (self.batch_size / sum_tour_length)
        return neg_associations, np.mean(tour_lengths)

    def gen_supernode(self, supernode_samples=None):
        if supernode_samples is None:
            supernode_samples = self.rbm.supernode_samples
        Log.info("Generating supernode")
        start = time.time()

        supernode_sizes = []
        supernode_matrix = []
        for _ in range(supernode_samples):
            hs = self.rbm.hard_samples(self.rbm.hidden_from_visible(self.train))
            supernode_matrix.append(hs)
            supernode_sizes.append(hs.shape[0])

        supernode_matrix = torch.cat(supernode_matrix, dim=0)
        supernode = Util.unique(supernode_matrix, self.rbm.gpu, True)
        end = time.time()
        Log.var(supernode_sizes=supernode_sizes, time=end - start)
        return supernode

    def sample_tour_length_distribution(self, sample_uniform):
        Log.info("Generating Tour Length Distribution")
        tour_limit = 1000
        tour_supernode_sizes = [1, 4, 7]
        num_samples = 150

        for supernode_samples in tour_supernode_sizes:
            supernode = self.gen_supernode(supernode_samples)
            for mini_batch_id in range(num_samples):
                history, total_tours, completed_tours, diverse_tours, tour_ccdf = self.run_tours(tour_limit, supernode,
                                                                                                 sample_uniform=sample_uniform)
                tour_length_dist = defaultdict(int)
                for h in history:
                    tour_length = len(h) - 1
                    tour_length_dist[tour_length] += 1

                SQLite().insert_dict(
                    TOUR_LENGTHS_TABLE,
                    Util.dictize(
                        name=self.rbm.name,
                        iter=self.iteration,
                        hidden=self.rbm.num_hidden,
                        sample_uniform=sample_uniform,
                        epoch=self.epoch,
                        mini_batch_id=mini_batch_id,
                        total_tours=total_tours,
                        completed_tours=completed_tours,
                        diverse_tours=diverse_tours,
                        supernode_size=supernode.hashes.shape[0],
                        supernode_samples=supernode_samples,
                        tour_lengths=Util.dict_to_json(tour_length_dist)
                    )
                )
                Log.info("Sampled supernode_samples=%s mini_batch_id=%s", supernode_samples, mini_batch_id)

    @staticmethod
    def get_state_hash(row):
        return tuple(row)

    def is_in_supernode(self, term_state, supernode):
        return self.rbm.gpu.tensor_converter(
            torch.eq(term_state[:, None, :], supernode.actual).min(dim=2)[0].max(dim=1)[0].sign())
