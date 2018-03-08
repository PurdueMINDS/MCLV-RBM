from numbers import Number

import numpy as np
import torch

from u.config import PIN, USE_DOUBLE, LOCAL


class Gpu:
    def __init__(self, gpu_id=None):
        self.use_gpu = torch.cuda.is_available() and not LOCAL
        self.gpu_id = int(gpu_id)
        if gpu_id is None:
            self.use_gpu = False

        if USE_DOUBLE:
            self.tensor_init = torch.DoubleTensor
            self.gpu_tensor_init = torch.cuda.DoubleTensor
            self.tensor_dt = torch.DoubleTensor
            self.tensor_converter = lambda self: self.double()
        else:
            self.tensor_init = torch.FloatTensor
            self.gpu_tensor_init = torch.cuda.FloatTensor
            self.tensor_dt = torch.FloatTensor
            self.tensor_converter = lambda self: self.float()

    def transfer(self, var):
        if self.use_gpu:
            with torch.cuda.device(self.gpu_id):
                if PIN:
                    var = var.pin_memory()
                var = var.cuda(device=self.gpu_id, async=PIN)
        return var

    def tensor(self, *args):
        if self.use_gpu:
            with torch.cuda.device(self.gpu_id):
                return self.gpu_tensor_init(*args)
        else:
            return self.tensor_init(*args)

    def pinned_transfer(self, var):
        if self.use_gpu:
            with torch.cuda.device(self.gpu_id):
                var = var.pin_memory()
                var = var.cuda(device=self.gpu_id, async=True)
        return var

    def from_numpy(self, var):
        var = torch.from_numpy(var)
        if (type(var) == torch.DoubleTensor and not USE_DOUBLE) or \
                (type(var) == torch.FloatTensor and USE_DOUBLE):
            raise Exception("")
        return self.transfer(var)

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """
        Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()

        Credits:
        Jan-Willem van de Meent (http://www.ccs.neu.edu/home/jwvdm/)
        Taken from: https://github.com/pytorch/pytorch/issues/2591
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                           dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            if isinstance(sum_exp, Number):
                return m + np.math.log(sum_exp)
            else:
                return m + torch.log(sum_exp)

    def log_sum_1_plus_exp(self, value):
        return torch.log(torch.exp(-value.clamp(min=0)) + torch.exp(value.clamp(max=0))).add(value.clamp(min=0))

    def ones(self, *args):
        return self.tensor_converter(self.transfer(torch.ones(*args)))

    def zeros(self, *args):
        return self.tensor_converter(self.transfer(torch.zeros(*args)))

    def float_range(self, *args):
        return self.tensor_converter(self.transfer(torch.arange(*args)))

    def long_range(self, *args):
        return self.transfer(torch.arange(*args)).long()