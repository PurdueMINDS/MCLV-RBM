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

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True

from util.config import SERVER_SQLITE_FILE, PLOT_OUTPUT_FOLDER
from util.sqlite import SQLite
from util.utils import Util


class PlotService:
    font_scale = 1.75
    line_styles = [
        [], [3, 6, 3, 6, 3, 18], [12, 6, 12, 6, 3, 6], [12, 6, 3, 6, 3, 6]
    ]

    @classmethod
    def plot(cls):
        result_set = lambda: [Res(row) for row in SQLite(SERVER_SQLITE_FILE).get_tour_data()]
        cls.compare_num_hidden_units(result_set())
        cls.compare_supernode_samples(result_set())

    @classmethod
    def compare_num_hidden_units(cls, result_set, k=1, epoch=100, supernode_samples=1):
        cls.init_ccdf_plot()

        plot_lines = dict()
        for row in result_set:
            if row.k == k and row.supernode_samples == supernode_samples:
                Util.put_or_add(plot_lines, row.hidden, row)

        plot_lines = sorted(plot_lines.values(), key=lambda r: int(r.hidden))
        max_returns = max([sum(r.completed_tours) for r in plot_lines])
        for i, plot_line in enumerate(plot_lines):
            x_ccdf, y_ccdf = plot_line.get_ccdf(max_returns)
            plt.plot(x_ccdf, y_ccdf, label="$n_H$ = %s" % plot_line.hidden, dashes=cls.line_styles[i])

        cls.finish_ccdf_plot('compare_num_hidden_units_%s_%s_%s.pdf' % (k, epoch, supernode_samples), asp=0.3)

    @classmethod
    def compare_supernode_samples(cls, result_set, k=1, epoch=100, hidden=32):
        cls.init_ccdf_plot()

        plot_lines = dict()
        for row in result_set:
            if row.k == k and row.hidden == hidden and row.supernode_samples in {1, 4, 7}:
                Util.put_or_add(plot_lines, row.supernode_samples, row)
        plt.clf()

        plot_lines = sorted(plot_lines.values(), key=lambda r: int(r.supernode_samples))
        max_returns = max([sum(r.completed_tours) for r in plot_lines])
        for i, plot_line in enumerate(plot_lines):
            x_ccdf, y_ccdf = plot_line.get_ccdf(max_returns)
            plt.plot(x_ccdf, y_ccdf,
                     label="$\\mathcal{S}^{(%s)}_{HN}$" % plot_line.supernode_samples, dashes=cls.line_styles[i])

        cls.finish_ccdf_plot('compare_rounds_%s_%s_%s.pdf' % (hidden, k, epoch), asp=0.6)

    @classmethod
    def init_ccdf_plot(cls):
        plt.clf()
        sns.set(style="white", font_scale=cls.font_scale)
        sns.set_style("ticks")

    @classmethod
    def finish_ccdf_plot(cls, filename, asp=0.4):
        plt.axes().set_yscale('log')
        plt.axes().set_xscale('log')
        plt.legend(loc='upper right')

        plt.axes().set_aspect(asp)

        plt.ylabel('$p(\\xi > k)$')
        plt.xlabel('Tour Length $(k)$')
        plt.tight_layout()

        fig = plt.gcf()
        sns.despine(fig, left=True)
        # fig.set_size_inches(13.0, 6.5)
        fig.savefig(PLOT_OUTPUT_FOLDER + filename, dpi=300, transparent=True, bbox_inches='tight')
        plt.clf()


class Res:
    def __init__(self, row):
        self.hidden = int(row['hidden'])
        self.k = int(row['k'])
        self.epoch = int(row['epoch']) - 15
        self.supernode_samples = int(row['supernode_samples'])

        self.supernode_size = self.make_list(row['supernode_size'])
        self.total_tours = self.make_list(row['total_tours'])
        self.completed_tours = self.make_list(row['completed_tours'])
        self.tour_lengths = Util.json_to_dict(row['tour_lengths'])
        self.tour_lengths = {int(k): int(v) for k, v in self.tour_lengths.items()}

    @classmethod
    def make_list(cls, input, trans=int):
        return [trans(input)]

    def __add__(self, res):
        self.supernode_size = self.supernode_size + res.supernode_size
        self.total_tours = self.total_tours + res.total_tours
        self.completed_tours = self.completed_tours + res.completed_tours
        sum_dict = defaultdict(int)
        for d in [self.tour_lengths, res.tour_lengths]:
            for k, v in d.items():
                sum_dict[k] += v
        self.tour_lengths = sum_dict
        return self

    def get_ccdf(self):
        max_tour_length = 1000
        self.tour_lengths[max_tour_length] = sum(self.total_tours) - sum(self.tour_lengths.values())

        sorted_data = []
        tl = sorted(self.tour_lengths.items(), key=lambda t: t[0])
        for (l, ct) in tl:
            sorted_data += [l for _ in range(ct)]
        sorted_data = np.array(sorted_data)
        steps = np.array(range(len(sorted_data))) / float(len(sorted_data))
        tail_index = np.argmax(sorted_data == max_tour_length) or len(sorted_data)
        sorted_data = sorted_data[:tail_index]
        steps = steps[:tail_index]
        steps = 1 - steps

        return sorted_data, steps


if __name__ == '__main__':
    PlotService.plot()
