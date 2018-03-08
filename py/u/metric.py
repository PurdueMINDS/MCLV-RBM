from collections import defaultdict

from pandas import DataFrame, np

from u.config import EPOCH_TRAINING_TABLE
from u.log import Log
from u.sqlite import SQLite


class SubMetric:
    def __init__(self):
        self.likelihoods = []
        self.reconstruction_errors = []
        self.free_energy = []
        self.partition_function = []

    def best_likelihood(self):
        valid = [i for i in self.likelihoods if i is not None]
        return np.max(valid) if len(valid) > 0 else None


class Metric:
    TRAIN = 'train'
    TEST = 'test'
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
    LOG_PARTITION = 'log_partition'
    PHASE = 'phase'
    EPOCH = 'epoch'
    NAME = 'name'
    GD_ANGLE = 'gradient_angle'
    GD_MAG = 'gradient_magnitude'
    LR = 'learning_rate'
    CCDF = 'ccdf'
    PARTITION_ESTIMATE = "partition_estimate"

    def __init__(self):
        self.train = SubMetric()
        self.test = SubMetric()

    def log_table(self, epoch_number, epoch_time, eval_time, tour_detail, gd_angle, gd_mag, lr, ccdf, mat_z_estimates,
                  name):
        if tour_detail is None:
            (completed_tours, total_tours, diverse_tours) = (-1, -1, -1)
        else:
            (completed_tours, total_tours, diverse_tours) = tour_detail
        df = defaultdict(dict)

        for (k, sm) in [(self.TRAIN, self.train), (self.TEST, self.test)]:
            d = {
                self.LIKELIHOOD: sm.likelihoods[-1]
                , self.RECON: sm.reconstruction_errors[-1]
                , self.FREE_ENERGY: sm.free_energy[-1]
                , self.BEST_LIKELIHOOD: sm.best_likelihood()
                , self.LOG_PARTITION: sm.partition_function[-1]
                , self.PHASE: k
                , self.EPOCH: epoch_number
                , self.NAME: name
            }
            if k == self.TRAIN:
                d[self.EPOCH_TIME] = epoch_time
                d[self.EVAL_TIME] = eval_time
                d[self.TOUR_DETAIL] = "%s/%s" % (completed_tours, total_tours)
                d[self.RETURN_RATIO] = "%0.2f" % (
                    0.0 if total_tours == 0 else 100.0 * completed_tours / total_tours
                )
                d[self.COMPLETED_TOURS] = completed_tours
                d[self.TOTAL_TOURS] = total_tours
                d[self.DIVERSE_TOURS] = diverse_tours
                d[self.GD_ANGLE] = gd_angle
                d[self.GD_MAG] = gd_mag
                d[self.LR] = lr
                d[self.CCDF] = ccdf
                d[self.PARTITION_ESTIMATE] = mat_z_estimates
            df[k] = d
            SQLite.insert_dict(EPOCH_TRAINING_TABLE, d)

        df = DataFrame(df)
        df = df.loc[
            [
                self.LIKELIHOOD, self.BEST_LIKELIHOOD,
                self.LOG_PARTITION, self.FREE_ENERGY, self.RECON,
                self.GD_ANGLE, self.GD_MAG,
                self.EPOCH_TIME, self.EVAL_TIME,
                self.TOUR_DETAIL, self.RETURN_RATIO, self.COMPLETED_TOURS, self.TOTAL_TOURS, self.DIVERSE_TOURS,
                self.CCDF, self.PARTITION_ESTIMATE,
                self.PHASE, self.EPOCH, self.LR
            ],
            [self.TRAIN, self.TEST]]
        Log.info("EPOCH INFO %s \n %s", epoch_number, df)
        return df

    @classmethod
    def log_lightweight(cls, completed_tours, total_tours, diverse_tours, current_learning_rate, ccdf, mat_z_estimate):
        d = {
            # cls.TOUR_DETAIL: "%s/%s" % (completed_tours, total_tours),
            #          cls.RETURN_RATIO: "%0.2f" % (0.0 if total_tours == 0 else 100.0 * completed_tours / total_tours),
            #          cls.COMPLETED_TOURS: completed_tours, cls.TOTAL_TOURS: total_tours,
            #          cls.DIVERSE_TOURS: diverse_tours, cls.LR: current_learning_rate,
            cls.PARTITION_ESTIMATE: mat_z_estimate,
            cls.CCDF: ccdf,
                     }
        s = ""
        for k,v in d.items():
            s+="%s:%s " % (k,v)
        return s
