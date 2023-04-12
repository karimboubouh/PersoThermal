import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch

from src.conf import EVAL_ROUND
from src.utils import load, verify_metrics


class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True, head_width=0.75 * height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plot_train_history(logs, metric='accuracy', measure="mean", info=None, plot_peer=None, save_fig=False):
    if isinstance(logs, str):
        logs = load(logs)
    # get correct metrics
    _metric = metric.lower()
    metric, measure = verify_metrics(metric, measure)
    # prepare data
    logs = [[ll[metric] for ll in lg] for lg in logs.values()]
    std_data = None
    if measure == "mean":
        data = np.mean(logs, axis=0)
    elif measure == "mean-std":
        if plot_peer is None:
            data = np.mean(logs, axis=0)
        else:
            print(f">>>>> Plotting chart for Peer({plot_peer})...")
            data = logs
        std_data = np.std(logs, axis=0)
    elif measure == "max":
        data = np.max(logs, axis=0)
    else:
        data = np.std(logs, axis=0)
    # plot data
    xlabel = 'Rounds'
    ylabel = f' {measure.capitalize()} {_metric.capitalize()}'
    title = f'{_metric.capitalize()} vs. No. of rounds'
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
    x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
    # Configs
    plt.grid(linestyle='dashed')
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )
    # Plot
    plt.plot(x, data)
    if std_data is not None:
        plt.fill_between(x, data - std_data, data + std_data, alpha=.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_fig:
        unique = np.random.randint(100, 999)
        plt.savefig(f"./out/EXP_{unique}.pdf")

    plt.show()


def meta_arrows(logs):
    FL = [x[0]['FL']['val_rmse'] for x in logs.values()]
    meta_rmse = [x[0]['test']['rmse'] for x in logs.values()]
    # create scatter plot, specifying marker size to be 40
    peers = range(len(logs))
    # plt.style.use('ggplot')
    plt.ylim(None, 0.06)
    plt.scatter(peers, FL, s=30, color="#D7D7D7", label="FL meta model")
    for i in peers:
        if FL[i] > 0.1 or meta_rmse[i] > 0.1:
            continue
        dy = meta_rmse[i] - FL[i]
        # print(f"X={i} | y={FL[i]} | meta={meta_rmse[i]}| dy={dy}")
        if dy > 0:
            up = plt.arrow(x=i, y=FL[i], dx=0, dy=dy, color="#F92D45", width=0.2, head_width=1, head_length=0.001)
        else:
            down = plt.arrow(x=i, y=FL[i], dx=0, dy=dy, color="blue", width=0.2, head_width=1, head_length=0.001)

    # display plot
    plt.grid(axis='y', color='grey', linestyle='dashed', linewidth=1, alpha=0.2)
    plt.tick_params(bottom=False, left=True)
    plt.ylabel(f"Test RMSE [Fahrenheit]", fontsize=14, labelpad=10)
    # ax.set_title(f'Title', fontdict={'fontsize': 16, 'fontweight': 'bold'}, pad=10)
    plt.xlabel("100 Homes of cluster $0$")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    h, l = plt.gca().get_legend_handles_labels()
    h.append(down)
    h.append(up)
    l.append('Personalized model (Improved)')
    l.append('Personalized model (Degraded)')
    plt.legend(h, l, handler_map={mpatches.FancyArrow: HandlerArrow()}, loc='upper left')  #
    plt.tight_layout()
    plt.show()


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def box_plot(logs):
    # colors = ['#142457', 'aqua', 'blue', '#FBBD06', '#F92D45', '#027FFF', '#51AEB0', '#E7E7E7', ]
    colors = ['#F92D45', '#51AEB0', '#FBBD06', '#142457', '#027FFF', '#49A52C', '#272727', '#E7E7E7']
    data = np.array([x[0]['train']['rmse'] for x in logs.values()])
    vp = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=True)
    labels = ['1 Step', '2 Steps', '3 Steps', '4 Steps']
    plt.xticks([1, 2, 3, 4], labels)

    for i, v in enumerate(vp['bodies']):
        v.set_facecolor(colors[i])
        v.set_edgecolor(colors[i])
        v.set_alpha(0.4)
        v.colorbar = "red"
    # for ax in plt:
    #     set_axis_style(ax, labels)
    plt.xlabel('Personalization steps', fontsize=12)
    plt.ylabel('RMSE error per home', fontsize=12)
    plt.grid(axis='y', color='grey', linestyle='dashed', linewidth=1, alpha=0.2)
    plt.tick_params(bottom=False, left=True)
    plt.show()


if __name__ == '__main__':
    # train_logs = load("FL_TensorFlow_1H_cluster_0_summer_1_432.pkl")
    # train_logs = load("FL_TensorFlow_30min_cluster_0_summer_1_432.pkl")
    # train_logs = load("FL_TensorFlow_30min_cluster_0_summer_2_432.pkl")
    # train_logs = load("FL_TensorFlow_30min_cluster_0_summer_4_432.pkl")
    # train_logs = load("FL_TensorFlow_15min_cluster_0_summer_1_432.pkl")
    train_logs = load("FL_TensorFlow_5min_cluster_0_summer_1_432.pkl")
    # train_logs = load("FL_TensorFlow_5min_cluster_0_summer_4_432.pkl")
    # plot_train_history(train_logs[0], metric='rmse', measure="mean-std")
    meta_arrows(train_logs[1])
    # box_plot(train_logs[1])
