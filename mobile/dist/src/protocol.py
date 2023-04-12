import pickle

POPULATE = 0
CONNECT = 1
DISCONNECT = 2
TRAIN_STEP = 3
TRAIN_STOP = 4
PREFERENCES = 5
DEVICE_LOGS = 6
CONNECT_TO_NEIGHBOR = 7
NEIGHBORS = 8
CALL_METHOD = 9
RETURN_METHOD = 10


def populate(info):
    return pickle.dumps({
        'mtype': POPULATE,
        'data': info,
    })


def preferences(pref):
    return pickle.dumps({
        'mtype': PREFERENCES,
        'data': pref,
    })


def connect(address, node_id):
    return pickle.dumps({
        'mtype': CONNECT,
        'data': {'address': address, 'id': node_id},
    })


def connect_to_neighbor(host, port):
    return pickle.dumps({
        'mtype': CONNECT_TO_NEIGHBOR,
        'data': {'host': host, 'port': port},
    }, protocol=pickle.HIGHEST_PROTOCOL)


def neighbors(nbrs):
    return pickle.dumps({
        'mtype': NEIGHBORS,
        'data': {'nbrs': nbrs},
    }, protocol=pickle.HIGHEST_PROTOCOL)


def disconnect(node_id):
    return pickle.dumps({
        'mtype': DISCONNECT,
        'data': {'id': node_id},
    })


def call_method(method, *args, **kwargs):
    return pickle.dumps({
        'mtype': CALL_METHOD,
        'data': {'method': method, 'args': args, 'kwargs': kwargs},
    }, protocol=pickle.HIGHEST_PROTOCOL)


def return_method(method, r):
    return pickle.dumps({
        'mtype': RETURN_METHOD,
        'data': {'method': method, 'return': r},
    }, protocol=pickle.HIGHEST_PROTOCOL)


def train_step(t, update):
    return pickle.dumps({
        'mtype': TRAIN_STEP,
        'data': {'t': t, 'update': update},
    })


def stop_train():
    return pickle.dumps({
        'mtype': TRAIN_STOP,
        'data': {},
    })


def log(typ, txt):
    return pickle.dumps({
        'mtype': DEVICE_LOGS,
        'data': {'typ': typ, 'txt': txt},
    })
