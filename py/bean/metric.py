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

from collections import defaultdict

from pandas import DataFrame, np

TRAIN = 'train'
TEST = 'test'
MISC = 'misc'
LIKELIHOOD = 'likelihood'
RECON = 'recon'
FREE_ENERGY = 'free_energy'
BEST_LIKELIHOOD = 'best_likelihood'
EPOCH_TIME = 'epoch_time'
EVAL_TIME = 'eval_time'
TOUR_DETAIL = 'tour_detail'
TOTAL_TOURS = 'total_tours'
COMPLETED_TOURS = 'completed_tours'
DIVERSE_TOURS = 'diverse_tours'
RETURN_RATIO = 'return_ratio'


class SubMetric:
    def __init__(self):
        self.likelihoods = []
        self.reconstruction_errors = []
        self.free_energy = []

    def best_likelihood(self):
        valid = [i for i in self.likelihoods if i is not None]
        return np.max(valid) if len(valid) > 0 else None


class Metric:
    def __init__(self):
        self.train = SubMetric()
        self.test = SubMetric()

    def get_table(self, _epoch_time, _eval_time, _tour_detail):
        if _tour_detail is None:
            (completed_tours, total_tours, diverse_tours) = (-1, -1, -1)
        else:
            (completed_tours, total_tours, diverse_tours) = _tour_detail
        df = defaultdict(dict)

        for (k, sm) in [(TRAIN, self.train), (TEST, self.test)]:
            df[k][LIKELIHOOD] = sm.likelihoods[-1]
            df[k][RECON] = sm.reconstruction_errors[-1]
            df[k][FREE_ENERGY] = sm.free_energy[-1]
            df[k][BEST_LIKELIHOOD] = sm.best_likelihood()
        df[MISC][EPOCH_TIME] = _epoch_time
        df[MISC][EVAL_TIME] = _eval_time
        df[MISC][TOUR_DETAIL] = "%s/%s" % (completed_tours, total_tours)
        df[MISC][RETURN_RATIO] = "%0.2f" % (
            0.0 if total_tours == 0 else 100.0 * completed_tours / total_tours
        )
        df[MISC][COMPLETED_TOURS] = completed_tours
        df[MISC][TOTAL_TOURS] = total_tours
        df[MISC][DIVERSE_TOURS] = diverse_tours

        df = DataFrame(df)
        df = df.loc[
            [LIKELIHOOD, RECON, FREE_ENERGY, BEST_LIKELIHOOD, EPOCH_TIME, EVAL_TIME, TOUR_DETAIL,
             RETURN_RATIO, COMPLETED_TOURS, TOTAL_TOURS, DIVERSE_TOURS],
            [TRAIN, TEST, MISC]]
        return df
