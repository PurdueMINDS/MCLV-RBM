import itertools
from enum import Enum

import numpy as np
import torch
from tqdm import tqdm

from u.config import GPU_LIMIT, TIME_TEST, LOCAL, NP_DT
from u.log import Log
from u.utils import Util


class PartitionEstimator:
    def __init__(self, rbm):
        """
        :type rbm: r.rbm.RBM
        """
        self.W, self.num_hidden, self.num_visible = rbm.W, rbm.num_hidden, rbm.num_visible
        self.rbm = rbm

    def partition(self):
        if self.num_hidden < self.num_visible: return self._partition_marginalize_h()
        return self._partition_marginalize_v()

    def _partition_marginalize_h(self):  # Training RBMs Eq(20)
        hidden_space = np.array(list(itertools.product([0, 1], repeat=self.num_hidden)))
        hidden_space = np.insert(hidden_space, 0, 1, 1)
        Z = 0
        for h in hidden_space:
            Z += self.hidden_partition(h)
        ZZ, _ = self.get_partition_and_gradient()
        return Z

    def _partition_marginalize_v(self):  # Training RBMs Eq(20)
        visible_space = np.array(list(itertools.product([0, 1], repeat=self.num_visible)))
        visible_space = np.insert(visible_space, 0, 1, 1)
        Z = 0
        for v in visible_space:
            Z += self.visible_partition(v)
        ZZ = np.exp(self.marginal_cuda(visible_space, self.NT.visible, self.RT.log_sum))
        return Z

    def visible_partition(self, v):  # Training RBMs Eq(20)
        vw = np.dot(v, self.W)
        z = 1 + np.exp(vw)
        z = z[1:].prod()
        z *= np.exp(vw[0])
        return z

    def hidden_partition(self, h):  # Training RBMs Eq(20)
        wh = np.dot(self.W, h)
        z = 1 + np.exp(wh)
        z = z[1:].prod()
        z *= np.exp(wh[0])
        return z

    class NT(Enum):
        hidden = 1
        visible = 2

    class RT(Enum):
        mean_log = 1  # Mean of the Logarithm
        log_sum = 2  # Logarithm of the Summation
        normalized = 3  # Normalized Probability Array, Logarithm of the sum
        un_normalized = 4  # UnNormalized Probability Array, Logarithm of the sum

    def marginal_cuda(self, x, nt: NT, rt: RT):  # Training RBMs Eq(20)
        # h samples x dimension
        if nt == self.NT.hidden:
            wx = torch.mm(x, self.W.t())
        else:
            wx = torch.mm(x, self.W)

        # wx -> samples x alt_dimension
        lg_energy = self.rbm.gpu.log_sum_1_plus_exp(wx)
        lg_energy[:, 0] = wx[:, 0]

        lg_marginal_probabilities = lg_energy.sum(dim=1)
        if rt == self.RT.mean_log:
            return lg_marginal_probabilities.mean()
        else:
            log_sum = self.rbm.gpu.log_sum_exp(lg_marginal_probabilities)
            if rt == self.RT.log_sum:
                return log_sum
            elif rt == self.RT.normalized:
                return torch.exp(lg_marginal_probabilities - log_sum), log_sum
            elif rt == self.RT.un_normalized:
                return lg_marginal_probabilities, log_sum

    def weighted_marginal(self, v, w_a, w_b, beta):  # Salakhutdinov Page 42 unnumbered
        wx_a = torch.matmul(v, w_a)
        wx_b = torch.matmul(v, w_b)

        # wx samples x alt_dimension
        ones = self.rbm.gpu.ones(wx_a.shape[0], wx_a.shape[1])

        m_a = ones + self.rbm.gpu.tensor_converter(torch.exp(wx_a.mul(1 - beta)))
        m_b = ones + self.rbm.gpu.tensor_converter(torch.exp(wx_b.mul(beta)))

        m_a = torch.log(m_a)
        m_b = torch.log(m_b)

        m_a[:, 0] = ((1 - beta) * wx_a)[:, 0]
        m_b[:, 0] = (beta * wx_b)[:, 0]

        m_a = m_a.sum(dim=1)
        m_b = m_b.sum(dim=1)
        return m_a + m_b

    def weighted_sample(self, v, w_a, w_b, beta, runs):
        if beta == 0:
            v_bias = w_a[:, 0]
            v = torch.bernoulli(torch.sigmoid(
                v_bias.expand(runs, v_bias.shape[0])
            ))
        else:
            h_a = torch.bernoulli(torch.sigmoid(
                torch.matmul(v, w_a).mul(1 - beta)
            ))
            h_a[:, 0] = 1
            h_b = torch.bernoulli(torch.sigmoid(
                torch.matmul(v, w_b).mul(beta)
            ))
            h_b[:, 0] = 1
            v = torch.bernoulli(torch.sigmoid(
                torch.matmul(h_a, torch.t(w_a)).mul(1 - beta) + torch.matmul(h_b, torch.t(w_b)).mul(beta)
            ))
        v[:, 0] = 1
        return v

    def ais_partition(self, annealing_runs=100):  # Russlan's notation
        if TIME_TEST:
            return 1.0
        w_b = self.rbm.W + 0

        w_a = self.rbm.W + 0
        w_a[1:, 1:] = 0

        fac = 1 if LOCAL else 100
        betas = np.linspace(0, 0.5, 5 * fac, endpoint=False)
        betas = np.append(betas, np.linspace(0.5, 0.9, 40 * fac, endpoint=False))
        betas = np.append(betas, np.linspace(0.9, 1.0, 100 * fac + 1, endpoint=True))

        Log.debug("Starting AIS\n")
        wts = self.inside_tour = self.rbm.gpu.zeros(annealing_runs)
        v = None
        for k in tqdm(range(len(betas) - 1), desc="AIS", file=self.rbm.get_tqdm_file()):
            beta = betas[k]
            beta_n = betas[k + 1]
            v = self.weighted_sample(v, w_a, w_b, beta, annealing_runs)
            wts += self.weighted_marginal(v, w_a, w_b, beta_n) - self.weighted_marginal(v, w_a, w_b, beta)
        z_a = torch.sum(torch.log(1 + torch.exp(w_a[1:, 0]))) + torch.sum(torch.log(1 + torch.exp(w_a[0, 1:])))

        wt_sum = self.rbm.gpu.log_sum_exp(wts)
        w_mean = wt_sum - np.log(wts.shape[0])
        z_est = z_a + w_mean

        wts = torch.exp(wts - wt_sum)
        h = self.rbm.hidden_from_visible(v)
        negative_association = (v * wts[:, None]).t().mm(h) / (wts.sum())
        Log.dvar(z_a=z_a, w_mean=w_mean, z_est=z_est)
        return z_est, negative_association

    def get_partition_and_gradient(self):
        if TIME_TEST:
            return 1.0, None
        nt = self.NT.hidden if self.num_hidden < self.num_visible else self.NT.visible
        num = self.num_hidden if nt else self.num_visible
        if num <= GPU_LIMIT:
            space = np.array(list(itertools.product([0, 1], repeat=num)))
            space = np.insert(space, 0, 1, 1)
            space = self.rbm.gpu.from_numpy(space.astype(NP_DT))
            log_vh, lg_partition, _ = self.log_gradient(space, nt)
        else:
            space1 = np.array(list(itertools.product([0, 1], repeat=num - GPU_LIMIT)))
            space2 = np.array(list(itertools.product([0, 1], repeat=GPU_LIMIT)))
            lg_partition = []
            log_vh = None

            for i in tqdm(range(space1.shape[0]), desc="PE", file=self.rbm.get_tqdm_file()):
                can_space = np.repeat(np.array([space1[i, :]]), space2.shape[0], axis=0)
                space = np.concatenate((can_space, space2), axis=1)
                space = np.insert(space, 0, 1, 1)
                space = self.rbm.gpu.from_numpy(space.astype(NP_DT))
                _log_vh, log_sum, _ = self.log_gradient(space, nt)

                lg_partition.append(log_sum)
                if log_vh is None:
                    log_vh = _log_vh
                else:
                    log_vh = torch.cat((log_vh, _log_vh), 0)
            lg_partition = Util.log_sum_exp(np.array(lg_partition))
            log_vh = self.rbm.gpu.log_sum_exp(log_vh, 0, True)
        lg_gradient = log_vh[0, :, :] - lg_partition
        if nt == self.NT.visible:
            lg_gradient = torch.t(lg_gradient)
        return lg_partition, lg_gradient

    def log_gradient(self, s, nt):
        lg_marginal_probabilities, log_sum = self.marginal_cuda(s, nt, self.RT.un_normalized)

        if nt == self.NT.hidden:
            a = self.rbm.visible_from_hidden(s)
        else:
            a = self.rbm.hidden_from_visible(s)

        a = torch.log(a)
        lg_marginal_probabilities = lg_marginal_probabilities[:, None]

        a = a + lg_marginal_probabilities
        ma = torch.max(a)
        a -= ma
        a = torch.exp(a)
        log_vh = torch.log(torch.mm(torch.t(a), s))
        log_vh += ma
        return log_vh[None, :, :], log_sum, lg_marginal_probabilities
