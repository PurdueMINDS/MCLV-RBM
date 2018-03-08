import time

import numpy as np
import torch
from tqdm import tqdm

from b.learning_schedule import Schedule
from b.method import Method
from o.cdk_optimizer import CDKOptimizer
from o.mclvk_optimizer import MCLVKOptimizer
from o.optimizer import Optimizer
from r.partition_estimator import PartitionEstimator
from u.config import NAME_TABLE, GPU_HARD_LIMIT, DEBUG, GRADIENT_DETAIL_TABLE, NP_DT, LOG_FOLDER, LOCAL
from u.log import Log
from u.metric import Metric
from u.sqlite import SQLite
from u.times import Times
from u.utils import Util


class RBM:
    def __init__(self, num_visible=None, num_hidden=None, W=None, learning_rate=0.1, reinf_learning_rate=0.1,
                 weight_decay=0.0, cdk=1,
                 momentum=0.0, warmup_epochs=1, mclvk=200, schedule=Schedule.RM10, reinf_plateau=10, max_epochs=1,
                 batch_size=100,
                 method=Method.MCLV, supernode_samples=1, mat_nd=4, mat_batch_size=None, mat_ssf=None, gpu=None):
        # Config
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cdk = cdk
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.method = method

        # MCLV Specific
        self.warmup_epochs = warmup_epochs
        self.mclvk = mclvk
        self.supernode_samples = supernode_samples

        # #### Specific
        self.reinf_learning_rate = reinf_learning_rate
        self.reinf_plateau = reinf_plateau

        # ##### Specific
        self.mat_nd = mat_nd
        self.mat_batch_size = mat_batch_size
        self.mat_ssf = mat_ssf

        # PCD Specific
        self.persistent_chains = None

        # House Keeping
        self.last_change = 0
        self.metric = Metric()
        self.gpu = gpu
        self.W = W

        # Name Generation
        self.name, self.name_dict, self.qualified_name = self.generate_name()
        self.tqdm_file = LOG_FOLDER + self.qualified_name + ".tqdm"

    def generate_name(self):
        name_dict = Util.dictize(
            method=self.method

            , num_hidden=self.num_hidden
            , num_visible=self.num_visible
            , learning_rate=self.learning_rate
            , schedule=self.schedule
            , momentum=self.momentum
            , weight_decay=self.weight_decay
            , cdk=self.cdk
            , max_epochs=self.max_epochs
            , batch_size=self.batch_size

            # MCLV Specific
            , warmup_epochs=self.warmup_epochs
            , mclvk=self.mclvk
            , supernode_samples=self.supernode_samples

            # '######## Specific
            , reinf_learning_rate=self.reinf_learning_rate
            , reinf_plateau=self.reinf_plateau

            # '######## Specific
            , mat_nd=self.mat_nd
            , mat_batch_size=self.mat_batch_size
            , mat_ssf=self.mat_ssf
        )

        name = Util.md5(str(name_dict) + str(time.time()))
        name_dict["name"] = name
        qualified_name = "_".join([str(i) for i in name_dict.values()])
        name_dict["qualified_name"] = qualified_name
        SQLite.insert_dict(NAME_TABLE, name_dict)
        Log.var(name_dict=name_dict)
        self.name, self.name_dict, self.qualified_name = name, name_dict, qualified_name
        return name, name_dict, qualified_name

    def init_b(self, train):
        """
            Initialize the weights based on training data.
        :param train:
        :return:
        """
        if self.W is None:
            self.W = np.random.uniform(-.1, 0.1, (self.num_visible, self.num_hidden)) / np.sqrt(
                self.num_visible + self.num_hidden)
            self.W = np.array(self.W, dtype=NP_DT)
            self.W = np.insert(self.W, 0, 0, axis=1)
            self.W = np.insert(self.W, 0, 0, axis=0)
            self.W = self.gpu.from_numpy(self.W.astype(NP_DT))
        eps = 1e-2
        p = train.mean(dim=0)
        p = torch.log((p + eps) / (1 - p))
        self.W[1:, 0] = p[1:]

    def analyze_gradients(self, train, method, k, batch_size):
        self.cdk = k
        self.mclvk = k
        self.batch_size = batch_size
        computed_z = self.metric.test.partition_function[-1]

        h1 = int(0.3 * (self.num_hidden + 1))
        h2 = int(0.6 * (self.num_hidden + 1))
        v1 = int(0.3 * (self.num_visible + 1))
        v2 = int(0.6 * (self.num_visible + 1))

        optimizer = self.get_optimizer(method, -1, -1, train)
        s1_arr = []
        s2_arr = []
        gradient_arr = []
        n_norm_arr = []
        p_norm_arr = []
        mclv_z_arr = []
        for sample in tqdm(range(100), desc=str(method), file=self.get_tqdm_file()):
            idx = self.gpu.from_numpy(np.random.choice(np.arange(train.shape[0]), self.batch_size, True)).long()
            v = train[idx]
            h = self.hidden_from_visible(v)
            hs = self.hard_samples(h)
            times = Times()
            gradient, n_norm, p_norm = optimizer.get_gradient(v, h, hs, sample, times)
            mclv_z = optimizer.mat_z_estimates[-1]

            s1 = gradient[v1, h1]
            s2 = gradient[v2, h2]
            s1_arr += [s1]
            s2_arr += [s2]
            n_norm_arr += [n_norm]
            p_norm_arr += [p_norm]
            mclv_z_arr += [mclv_z]
            gradient_arr += [gradient.unsqueeze(0)]

        gm = torch.cat(gradient_arr).mean(dim=0).squeeze()
        angles = []
        for i, (g, s1, s2, n_norm, p_norm, mclv_z) \
                in enumerate(zip(gradient_arr, s1_arr, s2_arr, n_norm_arr, p_norm_arr, mclv_z_arr)):
            g = g.squeeze()
            angle = Util.compute_angle(g, gm)
            SQLite.insert_dict(GRADIENT_DETAIL_TABLE,
                               Util.dictize(
                                   name=self.name,
                                   method=method,
                                   k=k,
                                   batch_size=batch_size,
                                   h1=h1,
                                   h2=h2,
                                   v1=v1,
                                   v2=v2,
                                   s1=s1,
                                   s2=s2,
                                   n_norm=n_norm,
                                   p_norm=p_norm,
                                   itr=i,
                                   angle=angle,
                                   computed_z=computed_z,
                                   mclv_z=mclv_z
                               ))
            angles += [angle]

    def fit(self, train, test, iteration=None):
        Log.info("Starting fit with opt_type=%s \n %s", self.name, self.name_dict)
        self.init_b(train)
        optimizer = Optimizer(self, self.batch_size, -1)

        if not DEBUG:
            self.evaluate(train, test, epoch_time=-1, epoch_number=-1, tour_detail=None,
                          optimizer=optimizer)

        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            idx_list = Util.shuffle(np.arange(train.shape[0]))
            optimizer = self.get_optimizer(self.method, epoch, iteration, train)
            time_sum = Times()
            mini_batch_count = int(train.shape[0] / self.batch_size)
            for j in tqdm(range(mini_batch_count), desc="minibatches", file=self.get_tqdm_file()):
                # Sequential Samples of size batch size :
                idx = self.gpu.from_numpy(idx_list[j * self.batch_size:(j + 1) * self.batch_size])
                v = train[idx]
                h = self.hidden_from_visible(v)
                hs = self.hard_samples(h)
                times = optimizer.optimize(v, h, hs, j)
                time_sum = time_sum.add_all(times)

            Log.var(time_sum=time_sum.compute())

            # Report Epoch Time
            epoch_end = time.time()
            self.evaluate(train, test, epoch_time=(epoch_end - epoch_start), epoch_number=epoch
                          , tour_detail=(optimizer.completed_tours, optimizer.total_tours, optimizer.diverse_tours)
                          , optimizer=optimizer)
        return self

    def get_optimizer(self, method, epoch, iteration, train):
        if Method.requires_warmup(method) and \
                (epoch >= self.warmup_epochs or epoch < 0):
            if epoch == self.warmup_epochs:
                Log.info("Warm up complete, switching to %s" % (method))

            if method == Method.MCLV:
                optimizer = MCLVKOptimizer(self, train, self.batch_size, epoch, iteration)
            else:
                raise NotImplemented()
        else:
            optimizer = CDKOptimizer(self, self.batch_size, epoch, persistent=method == Method.PCD)
        return optimizer

    @staticmethod
    def hard_samples(p):
        return torch.bernoulli(p)

    def hidden_from_visible(self, V):
        p = torch.mm(V, self.W).sigmoid()
        p[:, 0] = 1.0
        return p

    def visible_from_hidden(self, H):
        p = torch.mm(H, self.W.t()).sigmoid()
        p[:, 0] = 1.0
        return p

    def evaluate(self, data, test, epoch_time, epoch_number, tour_detail, optimizer: Optimizer):
        """
        1. Computes partition function over all states and over visible and test states and prints out
        2. Computes actual gradient and compares

        :param data:
        :param test:
        :param epoch_time:
        :param epoch_number:
        :param tour_detail:
        :param optimizer:
        :return:
        """
        eval_start = time.time()
        L, Lt, log_Z, free_energy, free_energy_t, gradient = self.evaluate_gradient_partition(data, test)

        gd_angle = gd_mag = 0
        if gradient is not None and optimizer.batch_scaled_vht is not None:
            computed_gradient = optimizer.batch_scaled_vht.div(self.batch_size)
            gd_angle = torch.dot(computed_gradient.view(-1), gradient.view(-1)) \
                       / (torch.norm(computed_gradient, 2) * torch.norm(gradient, 2))
            gd_mag = torch.norm(computed_gradient - gradient, 2) / ((self.num_hidden + 1) * (self.num_visible + 1))

        error_t = self.get_reconstruction_errors(test)
        error = self.get_reconstruction_errors(data)

        eval_end = time.time()

        self.metric.test.likelihoods.append(Lt)
        self.metric.test.free_energy.append(free_energy_t)
        self.metric.test.reconstruction_errors.append(error_t)
        self.metric.test.partition_function.append(log_Z)

        self.metric.train.likelihoods.append(L)
        self.metric.train.free_energy.append(free_energy)
        self.metric.train.reconstruction_errors.append(error)
        self.metric.train.partition_function.append(log_Z)

        eval_time = eval_end - eval_start
        self.metric.log_table(epoch_number, epoch_time, eval_time, tour_detail, gd_angle, gd_mag, optimizer.lr,
                              optimizer.tour_ccdf, optimizer.mat_z_estimates, self.name)

    def evaluate_gradient_partition(self, data, test):
        pe = PartitionEstimator(self)
        free_energy = pe.marginal_cuda(data, pe.NT.visible, pe.RT.mean_log)
        free_energy_t = pe.marginal_cuda(test, pe.NT.visible, pe.RT.mean_log)
        if self.num_hidden <= GPU_HARD_LIMIT:
            log_Z, lg_gradient = pe.get_partition_and_gradient()  # Get the actual partition function
            gradient = torch.exp(lg_gradient)
        else:
            log_Z, gradient = pe.ais_partition()  # Get the AIS partition function
        L = free_energy - log_Z
        Lt = free_energy_t - log_Z
        return L, Lt, log_Z, free_energy, free_energy_t, gradient

    def get_reconstruction_errors(self, data):
        h = self.hidden_from_visible(data)
        hs = self.hard_samples(h)
        recon = self.visible_from_hidden(hs)
        return np.sqrt(torch.pow(data - recon, 2).sum(dim=1).mean())

    def get_tqdm_file(self):
        return None if LOCAL else open(self.tqdm_file, 'a')
