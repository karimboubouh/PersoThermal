import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import style
from matplotlib.lines import Line2D
from pandas import DataFrame

from src.conf import EVAL_ROUND
from src.helpers import Map
from src.utils import verify_metrics, load

# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
# matplotlib.use('cairo')
# matplotlib.use('svg')
# matplotlib.use('MacOSX')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def plot_predictions(predictions, info=None):
    if predictions is None:
        return
    fig, ax = plt.subplots()
    if predictions.test is not None:
        df: DataFrame = predictions.in_temp['test']  # [:72]
        df['prediction'] = predictions.test
        if predictions.n_steps is not None:
            steps = len(predictions.n_steps)
            df['n_steps'] = np.nan
            df['n_steps'][:steps] = predictions.n_steps
        df.plot(ax=ax)
        ax.set_xlabel(info.xlabel)
        ax.set_ylabel(info.ylabel)
        ax.legend(["Test temperature", "Prediction", "n-steps Prediction"])
        plt.show()
    if predictions.n_steps is not None:
        steps = len(predictions.n_steps)
        df: DataFrame = predictions.in_temp['test'][:steps].copy()
        df['prediction'] = predictions.n_steps
        df.plot(ax=ax)
        ax.set_xlabel(info.xlabel)
        ax.set_ylabel(info.ylabel)
        ax.legend(["Test temperature", f"{steps}-steps Prediction"])
        plt.show()
    if predictions.train is not None:
        df: DataFrame = predictions.in_temp['train']
        df['prediction'] = predictions.train
        df.plot(ax=ax)
        ax.set_xlabel(info.xlabel)
        ax.set_ylabel(info.ylabel)
        ax.legend(["Train temperature", "Prediction"])
        plt.show()


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


def plot_many_train_history(logs, save_fig=False):
    colors = ['blue', 'orange', 'black', 'purple', 'red', 'grey', 'tan', 'pink', 'navy', 'aqua']

    # prepare data
    data = []
    std_data = []
    for log in logs:
        if isinstance(log["data"], str):
            log_data = load(log["data"])
        else:
            log_data = log["data"]
        ld = [[ll["val_rmse"] for ll in lg] for lg in log_data.values()]
        data.append(np.mean(ld, axis=0))
        std_data.append(np.std(ld, axis=0))
    # plot
    x = range(0, len(data[0]) * EVAL_ROUND, EVAL_ROUND)
    for i, d in enumerate(data):
        plt.plot(x, d, label=logs[i]["label"], color=colors[i])
        plt.fill_between(x, d - std_data[i] / 2, d + std_data[i] / 2, alpha=.2)
    # Configs
    xlabel = 'Rounds'
    ylabel = f' Mean Test RMSE'
    plt.grid(linestyle='dashed')
    plt.legend(loc="best", shadow=True)
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )

    if save_fig:
        unique = np.random.randint(100, 999)
        plt.savefig(f"./out/EXP_{unique}.pdf")

    plt.show()


def scatter_plot(homes, meta):
    colors = ['#027FFF', '#51AEB0', '#FBBD06', '#F92D45']  # , '#E7E7E7', '#272727', '#49A52C', '#027FFF'

    if homes is None:
        return
    # style.use('seaborn')
    style.use('default')
    # style.use('seaborn-whitegrid')
    homes_rmse = [h.test.rmse for h in homes.values()]
    homes_mae = [h.test.mae for h in homes.values()]
    meta_rmse = [h.test.rmse for h in meta.values()]
    meta_mae = [h.test.mae for h in meta.values()]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(homes_rmse, homes_mae, color='r', alpha=0.6, label='Centralized model', marker='o')
    ax.scatter(meta_rmse, meta_mae, color='b', alpha=0.6, label='Personalized model', marker='o')
    ax.set_xlabel('Test RMSE', fontsize=13)
    ax.set_ylabel('Test MAE', fontsize=13)
    # ax.set_title('Centralized vs Personalized Models')
    plt.legend()
    plt.grid(linestyle='dashed')
    plt.show()


def box_plot(cl, logs, scope="all", showfliers=False, title=None):
    if cl is None or logs is None:
        return
    # style.use("ggplot")
    fig, ax = plt.subplots(1, 1)
    # test_loss = np.array([i.test.loss for i in logs.values()])
    # test_loss = test_loss[~np.isnan(test_loss)]
    test_rmse = np.array([i.test.rmse for i in logs.values()])
    test_rmse = test_rmse[~np.isnan(test_rmse)]
    test_mae = np.array([i.test.mae for i in logs.values()])
    test_mae = test_mae[~np.isnan(test_mae)]
    # test_data = [test_loss, test_rmse, test_mae]
    test_data = [test_rmse, test_mae]
    # test_labels = ["Test Loss", "Test RMSE", "Test MAE"]
    test_labels = ["Test RMSE", "Test MAE"]
    if scope != "test":
        # train_loss = np.array([i.train.loss for i in logs.values()])  # if i.train.loss is not np.nan
        # train_loss = train_loss[~np.isnan(train_loss)]
        train_rmse = np.array([i.train.rmse for i in logs.values()])
        train_rmse = train_rmse[~np.isnan(train_rmse)]
        train_mae = np.array([i.train.mae for i in logs.values()])
        train_mae = train_mae[~np.isnan(train_mae)]
        # train_data = [train_loss, train_rmse, train_mae]
        train_data = [train_rmse, train_mae]
        # train_labels = ["Train Loss", "Train RMSE", "Train MAE"]
        train_labels = ["Train RMSE", "Train MAE"]
        data = test_data + train_data
        labels = test_labels + train_labels
        colors = ['red', 'blue', 'orange', 'tan']
    else:
        colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
        data = test_data
        labels = test_labels
    for d in data:
        if np.isnan(np.sum(d)):
            exit("Contain nan values")
    box = ax.boxplot(data, vert=0, patch_artist=True, showfliers=showfliers, labels=labels)
    # box = plt.boxplot(box_plot_data, vert=0, patch_artist=True, labels=['course1', 'course2', 'course3', 'course4'],
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # ax.scatter(cl.test["loss"], 1, c='red')
    # ax.text(cl.train["loss"][0], 4.4, "CL Test Loss", c='red')

    ax.scatter(cl.test["rmse"], 1, c='black')
    ax.text(cl.test["rmse"], 1.3, "CL Test RMSE", c="red")

    ax.scatter(cl.test["mae"], 2, c='black')
    ax.text(cl.test["mae"], 2.3, "CL Test MAE", c='blue')

    if scope != "test":
        # ax.scatter(cl.train["loss"][-1], 1, c='red')
        # ax.text(cl.train["loss"][0], 1.4, "CL Train Loss", c='red')
        ax.scatter(cl.train["rmse"][-1], 3, c='black')
        ax.text(cl.train["rmse"][-1], 3.3, "CL Train RMSE", c="orange")
        ax.scatter(cl.train["mae"][-1], 4, c='black')
        ax.text(cl.train["mae"][-1], 4.3, "CL Train MAE", c='tan')

    if title:
        plt.title(title)
    fig.show()
    plt.show()


def boxs(files: dict, showfliers=False):
    colors = ['#027FFF', '#51AEB0', '#FBBD06', '#F92D45']  # , '#E7E7E7', '#272727', '#49A52C', '#027FFF'
    data = {}
    plot_data = []
    # style.use("ggplot")
    fig, ax = plt.subplots(1, 1)
    i = 0
    for t, file in files.items():
        train_log, _, _, homes_logs, _ = load(file)
        data[t] = {'logs': [train_log, homes_logs]}
        test_rmse = np.array([i.test.rmse for i in homes_logs.values()])
        data[t]['test_rmse'] = test_rmse[~np.isnan(test_rmse)]
        plot_data.append(data[t]['test_rmse'])
        print(f"{t}: {train_log.test['rmse']}")
        ax.scatter(train_log.test["rmse"], i + 1, c=colors[i], marker="H", s=80, zorder=10, label=f"{t} CL model")
        # ax.text(train_log.test["rmse"], i + 1.3, "CL Test RMSE", c="red")
        i = i + 1
    # plot_data.reverse()
    labels = list(files.keys())
    # labels.reverse()
    # colors.reverse()
    box = ax.boxplot(plot_data, vert=0, patch_artist=True, showfliers=showfliers, labels=labels)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    custom = [Line2D([], [], marker='H', color='#E7E7E7', markeredgecolor='black', markersize=10, label="Server test RMSE"),
              mpatches.Patch(facecolor='#E7E7E7', edgecolor='black', label='Homes test RMSE')]
    plt.legend(handles=custom)
    # red_patch = mpatches.Polygon(np.array([[0,1]]), color='red', label='The red data')
    # plt.legend(handles=[red_patch])

    plt.xlabel("Test RMSE [Fahrenheit]", fontsize=13)
    plt.ylabel("Temporal abstractions", fontsize=13)
    # plt.legend()


    plt.show()


def plot_many(logs, metric='accuracy', measure="mean", info=None):
    logs_0 = load("collab_log_100_0_234.pkl")
    logs_2 = load("collab_log_100_2_108.pkl")
    logs_10 = load("collab_log_100_10_776.pkl")
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    data_0 = np.mean([[v[metric] for v in lo] for lo in logs_0.values()], axis=0)
    data_2 = np.mean([[v[metric] for v in lo] for lo in logs_2.values()], axis=0)
    data_10 = np.mean([[v[metric] for v in lo] for lo in logs_10.values()], axis=0)

    # plot data
    xlabel = 'Number of rounds'
    ylabel = f'Test Accuracy'
    title = f'{_metric.capitalize()} vs. No. of rounds'
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    x = range(0, len(data_0) * EVAL_ROUND, EVAL_ROUND)
    # , color=colors[i], label=mean[i][1], linestyle=line_styles[i]
    plt.plot(x, data_0, label="Skip local step")  # , '-x'
    plt.plot(x, data_2, label="2 local epochs")  # , '-x'
    plt.plot(x, data_10, label="10 local epochs")  # , '-x'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.legend(loc="lower right", shadow=True)
    plt.show()


def plot_manymore(exps, metric='accuracy', measure="mean", info=None, save=False):
    # Configs
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    xlabel = 'Rounds'
    ylabel = f' {measure.capitalize()} {_metric.capitalize()}'
    title = f'{_metric.capitalize()} vs. No. of rounds'
    if info is not None:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel(xlabel, fontsize=13)
    colors = ['green', 'blue', 'orange', 'black', 'red', 'grey', 'tan', 'pink', 'navy', 'aqua']
    # colors = ['black', 'green', 'orange', 'blue', 'red', 'grey', 'tan', 'pink', 'navy', 'aqua']
    line_styles = ['-', '--', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':']
    plt.grid(linestyle='dashed')
    plt.rc('legend', fontsize=12)
    plt.xticks(fontsize=13, )
    plt.yticks(fontsize=13, )
    std_data = None
    for i, exp in enumerate(exps):
        # Data
        logs = load(exp['file'])
        name = exp.get('name', "")
        if measure == "mean":
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        elif measure == "mean-std":
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
            std_data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        elif measure == "max":
            data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        else:
            data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
        plt.plot(x, data, color=colors[i], label=name, linestyle=line_styles[i])
        if std_data is not None:
            plt.fill_between(x, data - std_data, data + std_data, color=colors[i], alpha=.1)

    plt.legend(loc="lower right", shadow=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title(title)
    if save:
        unique = np.random.randint(100, 999)
        plt.savefig(f"../out/EXP_{unique}.pdf")
    plt.show()


def plot_clusters(data, y_km, kmeans, K):
    # plot the k clusters
    colors = ['blue', 'yellow', 'orange', 'lightgreen', 'purple', 'grey', 'tan', 'pink', 'navy', 'aqua']
    ids = {}
    for k in range(K):
        ids[k] = data[y_km == k]
        plt.scatter(
            data[y_km == k, 0], data[y_km == k, 1],
            s=50, c=colors[k],
            marker='v', edgecolor='black',
            label=f"Cluster {k + 1} ({len(ids[k])} homes)"
        )

    # plot the centroids
    plt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        s=150, marker='*',
        c='red', edgecolor='black',
        label='Centroids'
    )
    # plt.rcParams['figure.figsize'] = (6.4, 4.8)
    plt.rcParams['figure.dpi'] = 300
    plt.xlabel('Floor area')
    plt.ylabel('Age')
    plt.legend(scatterpoints=1)
    plt.rc('legend', fontsize=4)
    plt.grid()
    # plt.savefig('Homes clusters.pdf')
    plt.show()


def duplicate(arr, factor):
    return [item for item in arr for _ in range(factor)]


def plot_n_steps_day(r05min, r15min, r30min, r1hour, title="N-step prediction (5 epochs)"):
    # info
    info = Map({'xlabel': "Time period", 'ylabel': 'Temperature'})
    colors = ['blue', 'orange', 'green', 'black', 'grey', 'navy']
    # dataFrame
    df = pd.DataFrame()
    df['test'] = r05min["test"]
    df['r05min'] = r05min["recursive_pred"]
    df['r15min'] = duplicate(r15min["recursive_pred"], 3)
    df['r30min'] = duplicate(r30min["recursive_pred"], 6)
    df['r1hour'] = duplicate(r1hour["recursive_pred"], 12)
    # figure
    ax = df.plot.line(color=colors)
    ax.set_xlabel(info.xlabel)
    ax.set_ylabel(info.ylabel)
    ax.legend(["Test temperature", "Pred 05min", "Pred 15min", "Pred 30min", "Pred 1hour"])
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    result_1hour = load("Predictions_5E_1H_974.pkl")
    result_30min = load("Predictions_5E_30min_974.pkl")
    result_15min = load("Predictions_5E_15min_974.pkl")
    result_05min = load("Predictions_5E_5min_974.pkl")

    plot_n_steps_day(result_05min, result_15min, result_30min, result_1hour)
    exit()

    # plot_n_steps_day(result_30min, title="Resolution 30min")

    # plot_n_steps_day(result_15min, title="Resolution 15min")

    # plot_n_steps_day(result_05min, title="Resolution 5min")

    # result_05min = {
    #     'test': [],
    #     'direct_pred': [],
    #     'recursive_pred': [],
    # }
