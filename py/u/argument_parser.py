import argparse

from b.learning_schedule import Schedule
from b.method import Method
from u.phase import Phase


class CustomArgumentParser:
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description)

    def _flag(self, variable, help=None):
        help = variable if help is None else help
        self.parser.add_argument('--%s' % variable.replace("_", "-"), dest=variable, action='store_const',
                                 const=True, default=False,
                                 help=help)
        return self

    def _val(self, variable, default, help=None):
        help = variable if help is None else help
        self.parser.add_argument('--%s' % variable.replace("_", "-"), dest=variable, action='store',
                                 default=default,
                                 help=help)
        return self

    def _list(self, variable, default, help=None):
        help = variable if help is None else help
        self.parser.add_argument('--%s' % variable.replace("_", "-"), dest=variable, action='store',
                                 default=default, nargs='+',
                                 help=help)
        return self

    @staticmethod
    def parse_top_level_arguments():
        parser = CustomArgumentParser(description='Fit RBM to MNIST using different gradient estimators')

        # Config
        parser._flag('local', 'Enables local run')
        parser._flag('double', 'Use double and not float')
        parser._flag('debug', 'Debug mode')
        parser._flag('timetest', 'Timetest mode')
        parser._flag('log_tour', 'Log tours')
        parser._flag('no_discrete', 'Dont Make visible states discrete')

        # Base
        parser._val('base_folder', '/Users/mkakodka/Code/Research/RBM_V1/', 'Base Folder for all directory paths')
        parser._val('sqlite', 'results')
        parser._val('phase', 'DATA', str(Phase.__dict__))
        parser._val('runs', 1, 'Number of runs')
        parser._list('gpu_id', [0])
        parser._val('gpu_limit', 18)
        parser._val('gpu_hard_limit', 25)
        parser._val('batch_size_limit', 2000)
        parser._list('filename', 'temp_local')
        parser._val('parallelism', 10, 'Sets the number of OpenMP threads used for parallelizing CPU operations per GPU')

        # RBM Config General
        parser._val('warmup_epochs', 2)
        parser._val('total_epochs', 10)
        parser._val('weight_decay', 0.0)
        parser._val('momentum', 0.0)
        parser._val('supernode_samples', 1)
        parser._val('cdk_warmup', -1)

        # RBM Config Lists
        parser._list('method', ['MCLV'], str(Method.__dict__))
        parser._list('k', [1], "steps")
        parser._list('batch_size', [128])
        parser._list('learning_rate', [0.1])
        parser._list('schedule', ['EXP100'], str(Schedule.__dict__))
        parser._list('num_hidden', [16])

        # RBM Config '########
        parser._list('mat_nd', [1], '########')
        parser._list('mat_samples', [1000], '########')
        parser._list('mat_tour_limit', [100], '########')
        parser._val('mat_batch_size', None, '########')
        parser._val('mat_ssf', None, '########')

        return parser.parser.parse_args()
