import pickle

import torch

from o.mclvk_optimizer import MCLVKOptimizer
from r.rbm import RBM
from u.config import MODEL_FOLDER
from u.log import Log
from u.gpu import Gpu


class RBMUtil:
    @staticmethod
    def run_tours(rbm: RBM, data, sample_uniform):
        Log.info("Logging tours for the trained model")
        optimizer = MCLVKOptimizer(rbm, data, rbm.batch_size, -1, 1)
        optimizer.sample_tour_length_distribution(sample_uniform)

    @staticmethod
    def save_model(rbm: RBM, filename):
        W = rbm.W
        rbm.W = None
        gpu = rbm.gpu
        rbm.gpu = None
        object_file = MODEL_FOLDER + "%s.object" % filename
        tensor_file = MODEL_FOLDER + "%s.tensor" % filename
        with open(object_file, 'wb') as op, open(tensor_file, 'wb') as tp:
            pickle.dump(rbm, op)
            Log.info("Model dumped to %s" % object_file)
            torch.save(W.cpu(), tp)
            Log.info("Tensor dumped to %s" % tensor_file)
        rbm.W = W
        rbm.gpu = gpu
        return object_file, tensor_file

    @staticmethod
    def load_model(filename, gpu):
        object_file = MODEL_FOLDER + "%s.object" % filename
        tensor_file = MODEL_FOLDER + "%s.tensor" % filename
        with open(object_file, 'rb') as op, open(tensor_file, 'rb') as tp:
            rbm = pickle.load(op)
            Log.info("Model Loaded from %s" % object_file)
            # This loads into default GPU, need to ensure that the GPU is fixed using export CUDA_VISIBLE_DEVICES
            W = torch.load(tp)
            Log.info("Tensor Loaded from %s" % tensor_file)
        rbm.W = gpu.transfer(W)
        assert isinstance(rbm, RBM)
        return rbm

    @staticmethod
    def compute_likelihood(rbm: RBM, data, test):
        L, Lt, log_Z, _, _, _ = rbm.evaluate_gradient_partition(data, test)
        Log.var(L=L, Lt=Lt, log_Z=log_Z)
        return log_Z, L, Lt


if __name__ == '__main__':
    RBMUtil.load_model()