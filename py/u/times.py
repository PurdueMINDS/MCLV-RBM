import time
from collections import defaultdict

import numpy as np


class Times:
    def __init__(self):
        self.start = time.time()
        self.times = []
        self.events = []

    def add(self, event):
        self.times.append(time.time())
        self.events.append(event)

    def add_all(self, that):
        self.times += that.times
        self.events += that.events
        return self

    def compute(self):
        tmap = defaultdict(lambda : (0, np.inf, 0))
        for time, event in zip(self.times, self.events):
            t = time - self.start
            (s, mi, ma) = tmap[event]
            tmap[event] = (s + t, min(mi, t), max(ma, t))
            self.start = time
        return tmap
