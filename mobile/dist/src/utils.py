import _multiprocessing
import random
import socket
import time

from src.conf import TCP_SOCKET_BUFFER_SIZE

_multiprocessing.sem_unlink = None
import numpy as np
from kivymd.toast import toast


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


def create_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


# def mnist(path, binary=True):
#     try:
#         open(path, 'r')
#     except FileNotFoundError as e:
#         toast(str(e))
#         exit()
#     X_train, Y_train = joblib.load(path)
#     Y_train = Y_train.astype(int).reshape(-1, 1)
#     if binary:
#         # Extract 1 and 2 from train dataset
#         f1 = 1
#         f2 = 2
#         Y_train = np.squeeze(Y_train)
#         X_train = X_train[np.any([Y_train == f1, Y_train == f2], axis=0)]
#         Y_train = Y_train[np.any([Y_train == f1, Y_train == f2], axis=0)]
#         Y_train = Y_train - f1
#         Y_train = Y_train.reshape(-1, 1)
#     else:
#         Y_train = np.array([np.eye(1, 10, k=int(y)).reshape(10) for y in Y_train])
#     X_train = X_train / 255
#
#     return X_train, Y_train


def sample_data(dataset, num_items):
    all_idxs = [i for i in range(len(dataset.data))]
    mask = list(np.random.choice(all_idxs, num_items, replace=False))
    np.random.shuffle(mask)
    data = dataset.data[mask, :]
    targets = dataset.targets[mask]
    return Map({'data': data, 'targets': targets})


def fixed_seed(fixed=True, seed=10):
    if fixed:
        random.seed(seed)
        np.random.seed(seed)


def labels_set(dataset):
    try:
        labels = set(dataset.train_labels_set)
    except AttributeError:
        classes = []
        for b in dataset:
            classes.extend(b[1].numpy())
        labels = set(classes)

    return labels


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
    toast(f"{predicate} finished after {time.time() - start_time} seconds.")
    print(f"{predicate} finished after {time.time() - start_time} seconds.")
    return False


def get_node_conn_by_id(node, node_id):
    for conn in node.neighbors:
        if conn.neighbor_id == node_id:
            return conn
    return None


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def norm_squared(vi, vj):
    fvi = np.concatenate([x.ravel() for x in vi])
    fvj = np.concatenate([x.ravel() for x in vj])
    return np.linalg.norm(fvi - fvj) ** 2
