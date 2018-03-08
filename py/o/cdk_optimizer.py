from o.optimizer import Optimizer
from u.config import DISCRETE


class CDKOptimizer(Optimizer):
    def __init__(self, rbm, batch_size, epoch, persistent):
        super().__init__(rbm, batch_size, epoch)
        self.persistent = persistent

    def get_negative_associations(self, v, h, hs, pos_associations, mini_batch_id, times):
        if self.persistent and self.rbm.persistent_chains is not None:
            hs = self.rbm.persistent_chains
        for _ in range(self.rbm.cdk):
            v = self.rbm.visible_from_hidden(hs)
            if DISCRETE:
                v = self.rbm.hard_samples(v)
            h = self.rbm.hidden_from_visible(v)
            hs = self.rbm.hard_samples(h)
        neg_associations = self._get_vh_cuda(v, h)
        if self.persistent:
            self.rbm.persistent_chains = hs
        return neg_associations, times
