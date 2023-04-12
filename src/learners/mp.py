import numpy as np
import torch
from tqdm import tqdm

from src.conf import EVAL_ROUND
from src.ml import model_inference, evaluate_model
from src.p2p import Graph, Node
from src.utils import log

name = "Model Propagation (MP)"


def collaborate(graph: Graph, args):
    log("info", f"Initializing Model Propagation...")
    if args.mp == 1:
        log('error', f"Model Propagation (MP) via message passing is not implemented yet.")
        exit(0)
    # init peers parameters
    for peer in graph.peers:
        peer.execute(train_init)
    graph.join()

    log("info", f"Collaborative training for T = {graph.args.rounds} rounds")
    for peer in graph.peers:
        T = tqdm(range(graph.args.rounds), position=0, desc=f"{peer}")
        peer.execute(train_step, T)
    graph.join()

    # stop train
    log("info", f"Evaluating the output of the collaborative training.")
    for peer in graph.peers:
        peer.execute(train_stop)
    graph.join()
    log('info', f"Graph G disconnected.")

    # get collaboration logs
    collab_logs = {peer.id: peer.params.logs for peer in graph.peers}

    return collab_logs


def train_init(peer: Node):
    r = peer.evaluate(peer.inference, one_batch=True)
    peer.params.logs = [r]
    peer.params.exchanges = 0
    peer.params.models = {}
    peer.params.Wi = peer.similarity
    peer.params.D = sum(peer.params.Wi.values())
    peer.params.c = calculate_c(peer)
    peer.params.alpha = 0.5
    return


def train_step(peer: Node, t):
    T = t if isinstance(t, tqdm) or isinstance(t, range) else [t]
    for t in T:
        # Randomly select a neighbor
        link = np.random.choice(peer.neighbors)
        neighbor = link.neighbor
        # Exchange models
        peer.params.models[neighbor.id] = neighbor.get_model_params()
        neighbor.params.models[peer.id] = peer.get_model_params()
        # Update model
        update_weights(peer, evaluate=(t % EVAL_ROUND == 0))
    return


def train_stop(peer: Node):
    model_inference(peer, one_batch=True)
    peer.stop()


# ----- Custom functions ------------------------------------------------------

def update_weights(peer, evaluate=False):
    W = peer.params.Wi
    d = peer.params.D
    a = peer.params.alpha
    a_bar = (1 - a)
    c = peer.params.c
    loc_wi = peer.local_model.get_params()
    wjs = peer.params.models
    w_sum = np.sum([(W[j] / d) * wj for j, wj in wjs.copy().items()])
    wi = (a + a_bar * c) ** -1 * (a * w_sum + a_bar * c * loc_wi)
    peer.set_model_params(wi)
    if evaluate:
        t_eval = peer.evaluate(peer.inference, one_batch=True)
        peer.params.logs.append(t_eval)
    return


def calculate_c(peer):
    peer_data_size = len(peer.train.dataset)
    max_neighbors_data_size = max([len(nl.neighbor.train.dataset) for nl in peer.neighbors])
    return peer_data_size / max(peer_data_size, max_neighbors_data_size)
