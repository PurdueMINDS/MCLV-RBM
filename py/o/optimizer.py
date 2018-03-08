import numpy as np
import torch

from b.learning_schedule import Schedule
from r.partition_estimator import PartitionEstimator
from u.config import GPU_HARD_LIMIT
from u.times import Times
from u.gpu import Gpu


class Optimizer:
    def __init__(self, rbm, batch_size, epoch):
        """

        :type rbm: rbm.rbm.RBM
        """
        self.rbm = rbm
        self.batch_size = batch_size
        self.pe = PartitionEstimator(rbm)
        self.epoch = epoch

        self.total_tours = 0
        self.completed_tours = 0
        self.diverse_tours = 0
        self.tour_ccdf = None
        self.mat_z_estimates = []

        self.batch_scaled_vht = None
        self.lr = 0
        pass

    def optimize(self, v, h, hs, mini_batch_id):
        times = Times()
        gradient, _,_ = self.get_gradient(v, h, hs, mini_batch_id, times)
        if gradient is not None:
            self._apply_gradient(gradient)
        times.add('apply_gradient')
        return times

    def get_gradient(self, v, h, hs, mini_batch_id, times):
        pos_associations = self._get_vh_cuda(v, h)
        times.add('positive_associations')
        neg_associations, times = self.get_negative_associations(v, h, hs, pos_associations, mini_batch_id, times)
        times.add('negative_associations')
        gradient = self._compute_gradient(neg_associations, pos_associations)
        times.add('get_gradient')
        mag = None if neg_associations is None else torch.norm(neg_associations, 2)
        return gradient, mag, torch.norm(pos_associations, 2)

    def _apply_gradient(self, w_update):
        self.rbm.W += w_update
        self.rbm.last_change = w_update

    def _compute_gradient(self, neg_associations, pos_associations):
        if neg_associations is None:
            return None
        # Update weights.
        self.lr = Schedule.get_learning_rate(self.rbm.schedule, self.rbm.learning_rate, self.epoch)
        w_update = (
            ((pos_associations - neg_associations) / self.batch_size)
            - (self.rbm.weight_decay * self.rbm.W)
        ).mul(self.lr) + self.rbm.momentum * self.rbm.last_change

        self.batch_scaled_vht = neg_associations
        return w_update

    def _get_vh_cuda(self, v, h):
        v = self.rbm.gpu.transfer(v)
        h = self.rbm.gpu.transfer(h)
        return torch.matmul(torch.t(v), h)

    def get_negative_associations(self, v, h, hs, pos_associations, mini_batch_id, times):
        raise NotImplementedError()
