from enum import Enum, auto

import numpy as np


class _Schedule:
    def __init__(self, factor):
        """

        :param factor: Learning Rate should become 1/factor in 100 epochs
        """
        self.factor = factor

    def rm(self, lr, epoch):
        """
        Use step wise reductions.
        :param lr:
        :param epoch:
        :return:
        """
        return lr * 1 / np.ceil((1 + epoch) * self.factor / 100)

    def exp(self, lr, epoch):
        """
        Uses exponential reduction
        :param lr:
        :param epoch:
        :return:
        """
        x = np.exp(1 / 99 * np.log(self.factor))
        return lr * 1 / np.power(x, epoch)


class Schedule(Enum):
    RM10 = 1
    RM100 = 2
    RM500 = 3
    RM1000 = 4
    EXP10 = 5
    EXP100 = 6
    EXP500 = 7
    EXP1000 = 8

    @staticmethod
    def get_learning_rate(schedule, lr, epoch):
        return {
            Schedule.RM10: _Schedule(10).rm,
            Schedule.RM100: _Schedule(100).rm,
            Schedule.RM500: _Schedule(500).rm,
            Schedule.RM1000: _Schedule(1000).rm,
            Schedule.EXP10: _Schedule(10).exp,
            Schedule.EXP100: _Schedule(100).exp,
            Schedule.EXP500: _Schedule(500).exp,
            Schedule.EXP1000: _Schedule(1000).exp
        }[schedule](lr, epoch)


if __name__ == '__main__':
    for epoch in range(100):
        print(epoch, [Schedule.get_learning_rate(s, 1.0, epoch) for s in Schedule])
