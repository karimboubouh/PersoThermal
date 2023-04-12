from __future__ import annotations

import pickle
import socket
import struct
import time
from copy import deepcopy
from threading import Thread

import numpy as np

from src import protocol
import src.conf as C
from src.ecobee import read_ecobee_cluster, prepare_ecobee, evaluate_cluster_model, make_predictions, \
    make_n_step_predictions
from src.helpers import Map, timeit
from src.ml import get_params, set_params, train_for_x_batches, initialize_models, evaluate_model
from src.ml import model_fit, model_inference
from src.utils import log, create_tcp_socket, get_ip_address, save


class Node(Thread):

    def __init__(self, k, model, data, neighbors_ids, clustered, similarity, args: Map, params=None):
        super(Node, self).__init__()
        self.id = k
        self.mp = bool(args.mp)
        self.host = get_ip_address()
        self.port = C.PORT + k
        self.model = model
        self.local_model = model
        self.grads = None
        self.V = {}
        self.current_round = 0
        self.current_exec = None
        self.neighbors_ids = neighbors_ids
        self.neighbors = []
        self.in_neighbors = []
        self.clustered = clustered
        self.similarity = similarity
        self.dataset = data
        self.terminate = False
        # default params
        self.params = Map({
            'frac': args.frac,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'gar': args.gar,
            'D': sum(self.similarity.values()),
            'confidence': 1,
            'alpha': 0.9,
            'batches': args.batches,
        })
        # override params if provided
        if isinstance(params, Map):
            self.params = Map(dict(self.params, **params))
        # initialize networks
        self._init_server()

    def run(self):
        if self.mp:
            while not self.terminate:
                try:
                    conn, address = self.sock.accept()
                    if not self.terminate:
                        neighbor_conn = NodeConnection(self, address[1], conn)
                        neighbor_conn.start()
                        self.neighbors.append(neighbor_conn)
                        # self.in_neighbors.append(in_neighbor_conn)
                except socket.timeout:
                    pass
                except Exception as e:
                    log('error', f"{self}: Node Exception\n{e}")

            for neighbor in self.neighbors:
                neighbor.stop()
            self.sock.close()
        log('log', f"{self}: Stopped")

    def connect(self, neighbor: Node):
        try:
            if neighbor.id in [n.neighbor_id for n in self.neighbors]:
                log('log', f"{self}, neighbor {neighbor} already connected.")
                return True
            if self.mp:
                sock = create_tcp_socket()
                sock.settimeout(C.SOCK_TIMEOUT)
                sock.connect((neighbor.host, neighbor.port))
                neighbor_conn = NodeConnection(self, neighbor.id, sock)
                neighbor_conn.start()
                neighbor_conn.send(protocol.connect(sock.getsockname(), self.id))
                self.neighbors.append(neighbor_conn)
            else:
                slink = NodeLink(self, neighbor, None)
                dlink = NodeLink(neighbor, self, slink)
                slink.link = dlink
                self.neighbors.append(slink)
                neighbor.neighbors.append(dlink)

            return True
        except Exception as e:
            log('error', f"{self}: Can't connect to {neighbor} -- {e}")
            return False

    def disconnect(self, neighbor_conn: NodeConnection):
        if not neighbor_conn.terminate:
            neighbor_conn.send(protocol.disconnect(self.id))
            neighbor_conn.terminate = True
            if neighbor_conn in self.neighbors:
                self.neighbors.remove(neighbor_conn)
            log('log', f"{self} disconnected from {neighbor_conn.neighbor_name}")

    def stop(self):
        for neighbor in self.neighbors:
            self.disconnect(neighbor)
        self.terminate = True
        if self.mp:
            self.sock.close()

    @staticmethod
    def send(neighbor, msg):
        neighbor.send(msg)

    def broadcast(self, msg, active=None):
        active = self.neighbors if active is None else active
        for neighbor in active:
            self.send(neighbor, msg)

    def execute(self, func, *args):
        try:
            self.current_exec = Thread(target=func, args=(self, *args), name=func.__name__)
            # self.current_exec = Process(target=func, args=(self, *args), name=func.__name__)
            # self.current_exec.daemon = True
            self.current_exec.start()
        except Exception as e:
            log('error', f"{self} Execute exception: {e}")
            return None

    def fit(self, args, inference=True, one_batch=False):
        # train the model
        if one_batch:
            train_history = train_for_x_batches(self, batches=args.epochs, evaluate=False, use_tqdm=True)
            # set local model variable
            self.local_model = self.model
        else:
            # return Map({'train': None, 'test': None})
            train_history = model_fit(self)
            # set local model variable
            self.local_model = self.model
        # evaluate against a one batch or the whole inference dataset
        if inference:
            test_history = model_inference(self)
        else:
            test_history = None

        return Map({'train': train_history, 'test': test_history})

    def train_one_epoch(self, batches=1, evaluate=False):
        return train_for_x_batches(self, batches, evaluate, use_tqdm=False)

    def evaluate(self, one_batch=False, verbose=False):
        # return model_inference(self, batch_size=self.params.batch_size, one_batch=one_batch)
        return evaluate_model(self, one_batch=one_batch, batch_size=self.params.batch_size, verbose=verbose)

    def save_model(self):
        pass

    def get_neighbors(self):
        return self.neighbors

    def set_neighbors(self, neighbors, similarity=None):
        self.neighbors = neighbors
        if similarity:
            self.similarity = similarity

    def reset_neighbors(self, nodes, similarity):
        pass

    def get_weights(self):
        return deepcopy(self.model.state_dict())

    def get_model_params(self, named=False, numpy=False):
        return get_params(self.model, named=named, numpy=numpy)

    def set_model_params(self, params, named=False, numpy=False):
        return set_params(self.model, params, named=named, numpy=numpy)

    def set_weights(self, w):
        self.model.load_state_dict(deepcopy(w))

    def get_gradients(self):
        return self.grads

    def set_gradients(self, grads):
        pass

    def take_step(self):
        self.model.train()

    #  Private methods --------------------------------------------------------

    def _eval_sample(self, sample):
        pass

    def _init_server(self):
        if self.mp:
            self.sock = create_tcp_socket()
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(C.SOCK_TIMEOUT)
            self.sock.listen(C.TCP_SOCKET_SERVER_LISTEN)
            self.port = self.sock.getsockname()[1]
        else:
            self.sock = None

    # Special methods
    def __repr__(self):
        return f"Node({self.id})"

    def __str__(self):
        return f"Node({self.id})"


class NodeConnection(Thread):
    def __init__(self, node, neighbor_id, sock):
        super(NodeConnection, self).__init__()
        self.node = node
        self.sock = sock
        self.address = None
        self.neighbor_id = neighbor_id
        self.neighbor_name = f"Node({neighbor_id})"
        self.terminate = False

    def run(self):
        # Wait for messages from device
        while not self.terminate:
            try:
                (length,) = struct.unpack('>Q', self.sock.recv(8))
                buffer = b''
                while len(buffer) < length:
                    to_read = length - len(buffer)
                    buffer += self.sock.recv(4096 if to_read > 4096 else to_read)
                if buffer:
                    data = pickle.loads(buffer)
                    del buffer
                    if data and data['mtype'] == protocol.TRAIN_STEP:
                        self.handle_step(data['data'])
                    elif data and data['mtype'] == protocol.CONNECT:
                        self.handle_connect(data['data'])
                    elif data and data['mtype'] == protocol.DEVICE_LOGS:
                        self.handle_logs(data['data'])
                    elif data and data['mtype'] == protocol.DISCONNECT:
                        self.handle_disconnect()
                    else:
                        log('error', f"{self.node.name}: Unknown type of message: {data['mtype']}.")
            except pickle.UnpicklingError as e:
                log('error', f"{self.node}: Corrupted message : {e}")
            except socket.timeout:
                pass
            except struct.error:
                pass
            except Exception as e:
                self.terminate = True
                log('error', f"{self.node} NodeConnection <{self.neighbor_name}> Exception\n{e}")
        self.sock.close()
        log('log', f"{self.node}: neighbor {self.neighbor_name} disconnected")

    def send(self, msg):
        try:
            if self.terminate:
                log('log', f"{self} tries to send on terminated")
            length = struct.pack('>Q', len(msg))
            self.sock.sendall(length)
            self.sock.sendall(msg)
        except socket.error as e:
            self.terminate = True
            log('error', f"{self}: Socket error: {e}: ")
        except Exception as e:
            log('error', f"{self}: Exception\n{e}")

    def stop(self):
        self.terminate = True

    def handle_step(self, data):
        try:
            self.node.params.exchanges += 1
        except Exception:
            print(self)
            print(self.node)
            print(self.node.params)
            exit()
        if self.node.current_round <= data['t']:
            if data['t'] in self.node.V:
                self.node.V[data['t']].append((self.neighbor_id, data['update']))
            else:
                # TODO remove this line after simulations
                self.node.V[data['t']] = [(self.neighbor_id, data['update'])]
                # nb_homes = 207
                # self.node.V[data['t']] = [(self.neighbor_id, data['update'])] * nb_homes

    def handle_connect(self, data):
        self.neighbor_id = data['id']
        self.address = data['address']

    @staticmethod
    def handle_logs(data):
        log(data['typ'], data['txt'])

    def handle_disconnect(self):
        self.terminate = True
        if self in self.node.neighbors:
            self.node.neighbors.remove(self)

    #  Private methods --------------------------------------------------------

    def __repr__(self):
        return f"NodeConn({self.node.id}, {self.neighbor_id})"

    def __str__(self):
        return f"NodeConn({self.node.id}, {self.neighbor_id})"


class NodeLink:
    def __init__(self, node: Node, neighbor: Node, link: NodeLink = None):
        self.node = node
        self.neighbor: Node = neighbor
        self.link = link
        # kept for compatibility with NodeConnection
        self.terminate = False
        self.neighbor_id = neighbor.id
        self.neighbor_name = str(neighbor)

    def send(self, msg):
        if msg:
            data = pickle.loads(msg)
            if data and data['mtype'] == protocol.TRAIN_STEP:
                self.link.handle_step(data['data'])
            elif data and data['mtype'] == protocol.CONNECT:
                self.link.handle_connect(data['data'])
            elif data and data['mtype'] == protocol.DISCONNECT:
                self.link.handle_disconnect()
            else:
                log('error', f"{self.node.name}: Unknown type of message: {data['mtype']}.")
        else:
            log('error', f"{self.node.name}: Corrupted message.")

    def handle_step(self, data):
        self.node.params.exchanges += 1
        if self.node.current_round <= data['t']:
            if data['t'] in self.node.V:
                self.node.V[data['t']].append((self.neighbor_id, data['update']))
            else:
                self.node.V[data['t']] = [(self.neighbor_id, data['update'])]

    def handle_connect(self, data):
        self.neighbor_id = data['id']

    def handle_disconnect(self):
        self.terminate = True
        if self in self.node.neighbors:
            self.node.neighbors.remove(self)

    def stop(self):
        self.terminate = True

    #  Private methods --------------------------------------------------------

    def __repr__(self):
        return f"NodeLink({self.node.id}, {self.neighbor_id})"

    def __str__(self):
        return f"NodeLink({self.node.id}, {self.neighbor_id})"


class Graph:

    def __init__(self, peers, topology, test_ds, args):
        self.device = args.device
        self.peers = peers
        self.clusters = topology['clusters']
        self.similarity = topology['similarities']
        self.adjacency = topology['adjacency']
        self.test_ds = test_ds
        self.args = args

    @staticmethod
    # @measure_energy
    # @profiler
    @timeit
    def centralized_training(args, cluster_id=0, season='summer', resample=False, predict=False, n_ahead=1,
                             n_step_predict=False, hval=False, meta=True):
        log('warning', f'ML engine: {C.ML_ENGINE}')
        log('event', 'Centralized training ...')
        # load Ecobee dataset
        log('info', f'Loading processed data for cluster {cluster_id} and season: {season}...')
        data = read_ecobee_cluster(cluster_id, season, resample=resample)
        # prepare Ecobee data to generate timeseries dataset with n_input samples in history
        n_input = C.LOOK_BACK * C.RECORD_PER_HOUR
        n_features = len(C.DF_CLUSTER_COLUMNS)
        shape = (n_input, n_features)
        ds, info = prepare_ecobee(data[season], season, ts_input=n_input, n_ahead=n_ahead, batch_size=args.batch_size)
        log('info', f"Initializing {args.model} model.")
        model = initialize_models(args.model, input_shape=shape, n_outputs=n_ahead, nbr_models=1, same=True)[0]
        server = Node(0, model, ds, [], False, {}, args)
        log('info', f"Start server training on {len(server.dataset.Y_train)} samples ...")
        history = server.fit(args, one_batch=False, inference=True)
        if predict:
            log('event', f"Predicting test temperatures...")
            predictions = make_predictions(server, info, ptype="test")
        else:
            predictions = None
        if n_step_predict:
            log('event', f"Predicting n steps ahead temperatures using previous predictions ...")
            n_steps_predictions = make_n_step_predictions(server, period="1day", info=info)
            if predict:
                predictions["n_steps"] = n_steps_predictions.n_steps
                steps = len(n_steps_predictions.n_steps)
                result = {
                    'test': info.in_temp.test[:steps].values.flatten(),
                    'direct_pred': predictions.test[:steps],
                    'recursive_pred': n_steps_predictions.n_steps,
                }
                save(f"Predictions_{args.epochs}E_{C.TIME_ABSTRACTION}", result)

                print(f"Actual data:\n{info.in_temp.test[:steps].values.flatten()}")
                print(f"Direct Prediction:\n{predictions.test[:steps]}")
                print(f"Recursive Prediction:\n{n_steps_predictions.n_steps}")
        else:
            n_steps_predictions = None
        if hval:
            log('event', f"Evaluation of the learned model using individual home records")
            home_histories, meta_histories = evaluate_cluster_model(server.model, cluster_id, season=season,
                                                                    batch_size=args.batch_size, one_batch=True,
                                                                    resample=resample, meta=meta)
        else:
            home_histories = None
            meta_histories = None
        server.stop()

        return history, predictions, n_steps_predictions, home_histories, meta_histories

    # @measure_energy
    # @profiler
    @timeit
    def local_training(self, inference=True, one_batch=False):
        t = time.time()
        log('event', 'Starting local training ...')
        histories = dict()
        for peer in self.peers:
            if isinstance(peer, Node):
                nb = len(peer.dataset.Y_train)
                if one_batch:
                    log('info', f"{peer} is performing local training on {self.args.batch_size} out of {nb} samples.")
                else:
                    log('info', f"{peer} is performing local training on {nb} samples.")
            histories[peer.id] = peer.fit(self.args, inference=inference, one_batch=one_batch)
            # peer.stop()
        t = time.time() - t
        log("success", f"Local training finished in {t:.2f} seconds.")

        return histories

    def evaluation(self, dataset="test"):
        if dataset not in ["train", "val", "test", "inference"]:
            log("warning", f" unsupported dataset type, fallback to: test")
            dataset = "test"
        history = {}
        for peer in self.peers:
            if dataset == "train":
                history[peer.id] = peer.evaluate(peer.train)
            elif dataset == "val":
                history[peer.id] = peer.evaluate(peer.val)
            elif dataset == "test":
                history[peer.id] = peer.evaluate(peer.test)
            elif dataset == "inference":
                history[peer.id] = peer.evaluate(peer.inference)

        return history

    # @measure_energy
    # @profiler
    def collaborative_training(self, learner, args):
        t = time.time()
        log('event', f'Starting collaborative training using {learner.name} ...')
        collab_logs = learner.collaborate(self, args)
        t = time.time() - t
        log("success", f"Collaborative training finished in {t:.2f} seconds.")
        return collab_logs

    def join(self, r=None):
        t = time.time()
        if isinstance(self.peers[0], Node):
            name = self.peers[0].current_exec.name
        else:
            name = self.peers[0].current_exec
        for peer in self.peers:
            if isinstance(peer, Node):
                if peer.current_exec is not None:
                    peer.current_exec.join()
                    del peer.current_exec
                    peer.current_exec = None
            else:
                if peer.current_exec is not None:
                    peer.wait_method(peer.current_exec)
                    peer.current_exec = None

        t = time.time() - t
        if r is not None:
            log("log", f"Round {r}: {name} joined in {t:.2f} seconds.")
        else:
            log("success", f"{name} joined in {t:.2f} seconds.")

    def get_peers(self):
        return self.peers

    def show_similarity(self, ids=False, matrix=False):
        log('info', "Similarity Matrix")
        if matrix:
            print(self.similarity)
        else:
            for peer in self.peers:
                if ids:
                    s = list(peer.similarity.keys())
                else:
                    s = {k: round(v, 2) for k, v in peer.similarity.items()}
                log('', f"{peer}: {s}")

    def show_neighbors(self, verbose=False):
        log('info', "Neighbors list")
        for peer in self.peers:
            if isinstance(peer, Node):
                log('', f"{peer} has: {len(peer.neighbors)} neighbors.")
            else:
                log('', f"{peer} has: {len(peer.neighbors)} neighbors: {peer.neighbors}")
            if verbose:
                # log('', f"{peer} neighbors: {peer.neighbors}")
                log('',
                    f"{peer} has: {len(peer.train.dataset)} train samples / {len(peer.val.dataset)} validation samples "
                    f"/ {len(peer.test.dataset)} test samples / {len(peer.inference.dataset)} inference samples")

                iterator = iter(peer.train)
                x_batch, y_batch = next(iterator)
                log('', f"{peer} has: [{len(peer.train.dataset)}] {set(y_batch.numpy())}")
                print()

    def set_inference(self, args):
        # TODO check this
        # for peer in self.peers:
        #     peer.inference = inference_ds(peer, args)
        pass

    def PSS(self, peer: Node, k):
        nid = [n.neighbor_id for n in peer.neighbors]
        nid.append(peer.id)
        candidates = [p for p in self.peers if p.id not in nid]
        k = min(k, len(candidates))
        return np.random.choice(candidates, k, replace=False)

    def __len__(self):
        return len(self.peers)
