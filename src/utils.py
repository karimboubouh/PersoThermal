import argparse
import logging
import os
import pickle
import random
import socket
import struct
import fcntl
import time
from collections import Counter
from inspect import getframeinfo, currentframe
from itertools import combinations

from sklearn.preprocessing import MinMaxScaler


import netifaces
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial import distance
from termcolor import cprint

import src.conf as C
from src import conf
from src.helpers import Map

args: argparse.Namespace = None


def exp_details(arguments):
    print('Experimental details:')
    if C.ML_ENGINE.lower() == "tensorflow" and len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        print(f'    Training using      : GPU')
        print(f'    Default GPU Device  : {tf.test.gpu_device_name()}')
    else:
        print(f'    Training using      : CPU')
    print(f'    Model      : {arguments.model.upper()}')
    print(f'    Optimizer  : {arguments.optimizer}')
    print(f'    Epochs     : {arguments.epochs}')
    print(f'    Batch size : {arguments.batch_size}')
    print(f'    Time Abs   : {C.TIME_ABSTRACTION}')
    print('Collaborative learning parameters:')
    print(f'    Data size             : {"Unequal" if arguments.unequal else "Equal"} data size')
    print(f'    Test scope            : {arguments.test_scope}')
    print(f'    Number of peers       : {arguments.num_users}')
    print(f'    Rounds                : {arguments.rounds}')
    print(f'    Communication channel : {"TCP" if arguments.mp else "Shared memory"}')
    print(f'    Seed                  : {arguments.seed}')
    print(f'    ML engine             : {C.ML_ENGINE}')

    return


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rounds', type=int, default=500,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--model', type=str, default='LSTM',
                        help='model name: RNN, LSTM, DNN or BNN')
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of active neighbors')
    parser.add_argument('--gar', type=str, default='average',
                        help='Gradient Aggregation rule to use: \
                         average, median, krum, aksel')
    parser.add_argument('--epochs', type=int, default=4,
                        help="the number of local epochs: E")
    parser.add_argument('--batches', type=int, default=1,
                        help="the number of collaborative learning batches: E")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--mp', type=int, default=0,
                        help='Use message passing (MP) via sockets or shared \
                        memory (SM). Default set to MP. Set to 0 for SM.')
    parser.add_argument('--test_scope', type=str, default='global', help="test \
                        data scope (local, neighborhood, global)")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--verbose', type=int, default=2, help='verbose')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    global args
    args = parser.parse_args()
    return args


def load_conf(use_cpu=False):
    if C.ML_ENGINE.lower() == "tensorflow":
        tf.get_logger().setLevel(logging.ERROR)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        if use_cpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            print(f"List of used GPUs:\n {tf.config.list_physical_devices('GPU')}")
    # sys.argv = ['']
    global args
    args = args_parser()
    return Map(vars(args))


def fixed_seed(fixed=True):
    global args
    if fixed:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)


def log(mtype, message):
    global args
    # args = Map({'verbose': 2})
    title = True
    if not mtype:
        title = False
        mtype = log.old_type
    log.old_type = mtype
    if args.verbose > -2:
        if mtype == "result":
            if title:
                cprint("\r Result:  ", 'blue', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'blue', end=' ')
            cprint(str(message), 'blue')
            log.old_type = 'result'
            return
    if args.verbose > -1:
        if mtype == "error":
            if title:
                cprint("\r Error:   ", 'red', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'red', end=' ')
            cprint(str(message), 'red')
            log.old_type = 'error'
            return
        elif mtype == "success":
            if title:
                cprint("\r Success: ", 'green', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'green', end=' ')
            cprint(str(message), 'green')
            log.old_type = 'success'
            return
    if args.verbose > 0:
        if mtype == "event":
            if title:
                cprint("\r Event:   ", 'cyan', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'cyan', end=' ')
            cprint(str(message), 'cyan')
            log.old_type = 'event'
            return
        elif mtype == "warning":
            if title:
                cprint("\r Warning: ", 'yellow', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'yellow', end=' ')
            cprint(str(message), 'yellow')
            log.old_type = 'warning'
            return
    if args.verbose > 1:
        if mtype == "info":
            if title:
                cprint("\r Info:    ", attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", end=' ')
            cprint(str(message))
            log.old_type = 'info'
            return
    if args.verbose > 2:
        if mtype not in ["info", "warning", "event", "success", "error", "result"]:
            if title:
                cprint("\r Log:     ", 'magenta', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", end=' ')
            cprint(str(message))
            log.old_type = 'log'


def elog(message, mtype="log"):
    frameinfo = getframeinfo(currentframe().f_back)
    filename = frameinfo.filename.split('/')[-1]
    linenumber = frameinfo.lineno
    info = 'File: %s, line: %d' % (filename, linenumber)
    log(mtype, message)
    exit()


# def partition(x, nb_nodes, cluster_data=True, method="random", random_state=None):
#     M = x.shape[0]
#     print(M)
#     exit(0)
#     if cluster_data:
#         # method: kmeans or random.
#         gm = GaussianMixture(nb_nodes, init_params=method, random_state=random_state)
#         gm.fit(x)
#         labels = gm.predict(x)
#         print(type(labels))
#         groups = [[x[labels == i], y[labels == i]] for i in range(nb_nodes)]
#     else:
#         shuffled_ids = np.random.permutation(M)
#         s = M // nb_nodes
#         groups = [[x[shuffled_ids][i * s:(i + 1) * s], y[shuffled_ids][i * s:(i + 1) * s]] for i in range(nb_nodes)]
#
#     return groups


def cluster_peers(nb_nodes, k):
    shuffled_ids = np.random.permutation(nb_nodes)
    multi = np.random.multinomial(nb_nodes, np.ones(k) / k, size=10)
    s = None
    for m in multi:
        if (m < 2).sum() == 0:
            s = m
            break
    if s is None:
        exit('Error: cannot cluster peers.')
    clusters = {}
    step = 0
    for i in range(k):
        clusters[i] = shuffled_ids[step: step + s[i]]
        step += s[i]
    return clusters


def similarity_matrix(mask, clusters, rho=0.2, data=None):
    """Compute the similarity matrix randomly or according to users data"""
    if isinstance(mask, int):
        randm = True
        nb_nodes = mask
    else:
        randm = False
        nb_nodes = len(mask)
    combi = combinations(range(nb_nodes), 2)
    pairs = []
    for c in combi:
        for cluster in clusters.values():
            if np.in1d(c, cluster).sum() == 2:
                pairs.append(c)
    similarities = np.zeros((nb_nodes, nb_nodes))
    # calculate similarity matrix
    for p in pairs:
        if randm:
            similarities[p] = similarities.T[p] = np.random.uniform(-1, 1)
        else:
            mask1 = np.sort(mask[p[0]])
            mask2 = np.sort(mask[p[1]])
            vec1 = data.targets[mask1].numpy()
            vec2 = data.targets[mask2].numpy()
            similarities[p] = similarities.T[p] = 1 - distance.cosine(vec1, vec2)
    # apply similarity threshold
    if rho is not None:
        sim_vals = similarities.copy()
        similarities[similarities < rho] = 0
        for i, sim in enumerate(similarities):
            if np.all(np.logical_not(sim)):
                if len(sim) > 0:
                    neighbor_id = random.choice(range(len(sim)))
                    sim[neighbor_id] = round(abs(sim_vals[i][neighbor_id]), 2)
                    log('warning', f"Node({i}) was attributed one peer: Node({neighbor_id}) with W={sim[neighbor_id]}")
                else:
                    log('error', f"The generated Random Graph is disconnected [s={rho}], Node({i}) has 0 neighbors.")
                    exit(0)

    # get adjacency matrix
    adjacency = similarities != 0

    return adjacency, similarities


def node_topology(i, topology):
    similarity = topology['similarities'][i]
    similarity = {key: value for key, value in enumerate(similarity) if value != 0}
    neighbors_ids = [j for j, adj in enumerate(topology['adjacency'][i]) if bool(adj) is True]
    return neighbors_ids, similarity


def verify_metrics(_metric, _measure):
    _metric = _metric.lower()
    if _metric.lower() not in ['accuracy', 'loss', 'rmse', 'mape', 'mae']:
        log("error", f"Unknown metric: {_metric}")
        log("", f"Set metric to default: accuracy")
        metric = f"{conf.DEFAULT_VAL_DS}_loss"
    else:
        metric = f"{conf.DEFAULT_VAL_DS}_{_metric}"

    if _measure not in ['mean', 'mean-std', 'max', 'std']:
        log("error", f"Unknown {_metric} measure: {_measure}")
        log("", f"Set measure to default: mean")
        measure = "mean"
    else:
        measure = _measure
    return metric, measure


def fill_history(a):
    lens = np.array([len(item) for item in a.values()])
    log('info', f"Number of rounds performed by each peer:")
    log('', f"{lens}")
    ncols = lens.max()
    last_ele = np.array([a_i[-1] for a_i in a.values()])
    out = np.repeat(last_ele[:, None], ncols, axis=1)
    mask = lens[:, None] > np.arange(lens.max())
    out[mask] = np.concatenate(list(a.values()))
    out = {k: v for k, v in enumerate(out)}
    return out


def active_peers(peers, frac):
    m = max(int(frac * len(peers)), 1)
    return np.random.choice(peers, m, replace=False)


def wait_until(predicate, timeout=2, period=0.2, *args_, **kwargs):
    start_time = time.time()
    mustend = start_time + timeout
    while time.time() < mustend:
        if predicate(*args_, **kwargs):
            return True
        time.sleep(period)
    log("error", f"{predicate} finished after {time.time() - start_time} seconds.")
    return False


def create_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, C.TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, C.TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def get_node_conn_by_id(node, node_id):
    for conn in node.neighbors:
        if conn.neighbor_id == node_id:
            return conn
    return None


def ip4_addresses(inter="all"):
    if inter == "all":
        ip4_list = []
        for interface in netifaces.interfaces():
            if netifaces.AF_INET in netifaces.ifaddresses(interface):
                ip4_list.append({interface: netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']})
        return ip4_list
    else:
        try:
            return netifaces.ifaddresses(inter)[netifaces.AF_INET][0]['addr']
        except ValueError as e:
            print(f"Error: {e}\nSelect one of these interfaces {netifaces.interfaces()}")
        except KeyError:
            print(f"Error: Network interface {inter} has no IPv4 address.")


def get_ip_address(inter=C.NETWORK_INTERFACE):
    if inter:
        return ip4_addresses(inter)
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(10)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())
        finally:
            s.close()


def labels_set(dataset):
    try:
        labels = set(dataset.train_labels_set)
    except AttributeError:
        classes = []
        for b in dataset:
            classes.extend(b[1].numpy())
        labels = set(classes)

    return labels


def save(filename, data):
    unique = np.random.randint(100, 999)
    filename = f"./out/{filename}_{unique}.pkl"
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        print("Writing to file", filename)
    return


def load(filename):
    with open(f"./out/{filename}", 'rb') as fp:
        return pickle.load(fp)


def norm_squared(vi, vj):
    if C.ML_ENGINE.lower() == "tensorflow":
        fvi = np.concatenate([x.ravel() for x in vi])
        fvj = np.concatenate([x.ravel() for x in vj])
        return np.linalg.norm(fvi[0] - fvj[0]) ** 2
    else:
        fvi = np.concatenate([x.ravel() for x in vi])
        fvj = np.concatenate([x.ravel() for x in vj])
        return np.linalg.norm(fvi - fvj) ** 2


def replace_many_zeros_columns(df, name):
    for c in df.columns:
        if len(df[c].dropna().values) > 0:
            v = df[c].dropna().values[0]
            df[c].fillna(v, inplace=True)
        else:
            print(f"replacing empty column {c} with values from in_temp column")
            if c in ['in_cool', 'in_heat']:
                df[c] = df['in_temp']
            else:
                log('error', f"Cannot fill the values of column {c} having not a single value in file {name}")
                exit()
    return df


def nb_pred_steps(pred, period: str):
    time_abs = [None, "5min", "15min", "30min", "1H"]
    log("info", f"C.TIME_ABSTRACTION={C.TIME_ABSTRACTION} || C.RECORD_PER_HOUR={C.RECORD_PER_HOUR}")
    if period == "1hour" and C.TIME_ABSTRACTION in time_abs:
        steps = C.RECORD_PER_HOUR
    elif period == "1day" and C.TIME_ABSTRACTION in time_abs:
        steps = 24 * C.RECORD_PER_HOUR
    elif period == "1week" and C.TIME_ABSTRACTION in time_abs:
        steps = 7 * 24 * C.RECORD_PER_HOUR
    elif period in ["test", "all"]:
        steps = len(pred)
    else:
        steps = None

    return steps


def get_homes_metadata(home_ids, filename="meta_data.csv", normalize=False):
    meta_file = os.path.join(C.DATA_DIR, filename)
    df = pd.read_csv(meta_file, usecols=C.META_CLUSTER_COLUMNS)
    meta = df.to_numpy()
    _, indices, _ = np.intersect1d(meta[:, 0], list(home_ids), return_indices=True)
    meta_abs = meta[indices][:, 1:]
    meta_ids = meta[indices][:, 0]

    if normalize:
        scaler = MinMaxScaler()
        meta_abs = scaler.fit_transform(meta_abs)

    return {meta_ids[i]: meta_abs[i] for i in range(len(meta_ids))}
