from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True

from u.config import SERVER_SQLITE_FILE, PLOT_OUTPUT_FOLDER, DATA_FOLDER
from u.sqlite import SQLite
from u.utils import Util


class PlotService:
    font_scale = 1.75
    line_styles = [
        [], [3, 6, 3, 6, 3, 18], [12, 6, 12, 6, 3, 6], [12, 6, 3, 6, 3, 6]
    ]
    line_styles2 = {
        'Method.CD': '-', 'Method.MCLV': ':', 'Method.PCD': '--'
    }
    colors = {
        'Method.CD': 'C0', 'Method.MCLV': 'C1', 'Method.PCD': 'C2'
    }
    VERSION = 100  # 50 has tour information, Models were trained in 25

    @classmethod
    def plot(cls):
        # cls.do_bar_graphs()
        result_set = lambda: [Res(row) for row in SQLite(SERVER_SQLITE_FILE).execute_query(
            "select b.method, coalesce(b.discrete,'True') discrete, \
                                    b.mclvk mclvk, b.cdk cdk ,  a.* from FINAL_EXTRA_LONG_TOUR_LENGTHS_NEW a \
                                    inner join NAME_DETAILS b on a.config = b.name \
                                    where version = %s;" % cls.VERSION)]
        for k in [1, 10]:
            for uniform in [False, True]:
                cls.compare_methods(result_set(), 32, mthds=['Method.CD', 'Method.MCLV', 'Method.PCD'], k=k,
                                    uniform=uniform)
                # cls.compare_methods(result_set(), 32, mthds=['Method.CD', 'Method.MCLV'], k=k, uniform=uniform)
                cls.compare_methods(result_set(), 32, mthds=['Method.CD'], k=k, uniform=uniform)
        # cls.compare_num_hidden_units(result_set())
        # cls.compare_supernode_samples(result_set())

    @classmethod
    def compare_methods(cls, result_set, h=25, mthds=[], uniform=True, k=1):
        cls.init_ccdf_plot()

        plot_lines = dict()
        for row in result_set:
            if row.hidden == h and row.method in set(mthds) and row.uniform == uniform \
                    and (row.cdk == k or row.mclvk == k):
                Util.put_or_add(plot_lines, row.method, row)

        if len(plot_lines) > 0:
            plot_lines = sorted(plot_lines.values(), key=lambda r: int(r.hidden))
            max_returns = max([sum(r.completed_tours) for r in plot_lines])
            for i, plot_line in enumerate(plot_lines):
                x_ccdf, y_ccdf = plot_line.get_ccdf(max_returns)
                print("%s, %s,%s,%s, %s" % (uniform, k, h, plot_line.method, np.min(y_ccdf[x_ccdf == 1])))
                plt.plot(x_ccdf, y_ccdf, label="%s" % plot_line.method.replace("Method.", "")
                         , linestyle=cls.line_styles2[plot_line.method], color=cls.colors[plot_line.method]
                         )

            cls.finish_ccdf_plot('compare_methods_%s-%s_%s-Nodes_UniformStarts-%s.pdf' % (str(mthds), k, h, uniform),
                                 asp=0.5)

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
    def do_bar_graphs(cls):
        sns.set(style="whitegrid", font_scale=cls.font_scale)
        long_labels = np.loadtxt(DATA_FOLDER + 'long_labels.data').astype('int')
        short_labels = np.loadtxt(DATA_FOLDER + 'short_labels.data').astype('int')
        df = defaultdict(dict)
        for labels, type in zip([short_labels, long_labels], ['Short Tours', 'Long Tours']):
            for l in labels:
                idx = type + str(l)
                df['Type'][idx] = type
                df['label'][idx] = l
                if idx in df['count']:
                    df['count'][idx] += 1
                else:
                    df['count'][idx] = 1
        df = pd.DataFrame(df)
        g = sns.factorplot(x="label", y="count", hue="Type", data=df,
                           aspect=2, kind="bar", palette="muted", legend=False)
        g.despine(left=True)
        g.set_ylabels("Number of Tours")
        g.set_xlabels("Class Label of the Starting State")
        # g.set_size_inches(10.0, 6.5)
        plt.legend(loc='best')
        g.savefig(PLOT_OUTPUT_FOLDER + "short_long_dist_bar.pdf", dpi=100)

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
        self.uniform = int(row['sample_uniform'])
        self.cdk = int(row['cdk'])
        self.mclvk = int(row['mclvk'])
        self.epoch = int(row['epoch']) - 15
        self.supernode_samples = int(row['supernode_samples'])
        self.method = row['method']
        self.discrete = row['discrete']

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

    def get_ccdf(self, max_returns):

        # unfinished = sum(self.total_tours) - sum(self.completed_tours)
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
