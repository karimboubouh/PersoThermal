from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm

from src.conf import RECORD_RATE
from src.p2p import Graph
from src.utils import log

name = "Average-based Collaborative Learner"
# name = "Gradient Averaging Collaborative Learner"



def collaborate(graph: Graph, device='cpu'):
    # initialize history holder
    history = dict.fromkeys(range(len(graph.peers)))
    for k in history.keys():
        history[k] = []

    # setup algorithm parameters
    for peer in graph.peers:
        peer.params.models = []

    # prepare tqdm
    rounds = tqdm(range(graph.args.rounds))
    log("info", f"Collaborative training for {graph.args.rounds} rounds")
    for epoch in rounds:
        # Randomly activate a peer
        peer = np.random.choice(graph.peers)
        if peer.clustered:
            # exchange with peer's cluster
            peer.params.models = []
            for neighbor in peer.neighbors:
                peer.params.models.append(neighbor.model)
            # Update model
            average_weights(peer)
        else:
            # Randomly select a neighbor
            neighbor = np.random.choice(peer.neighbors)
            # Exchange models
            peer.params.models = [neighbor.model]
            neighbor.params.models = [peer.model]
            # Update models
            average_weights(peer)
            average_weights(neighbor)

        # Evaluate all models every RECORD_RATE
        if (epoch + 1) % RECORD_RATE == 0:
            run_evaluation(graph, history)
            rounds.set_postfix({**{'peer': peer}, **history[peer.id][-1]})

    for peer in graph.peers:
        print(peer, peer.model.evaluate(peer.test))

    return history


def average_weights(peer):
    """Returns the average of the weights."""
    wi = deepcopy(peer.model.state_dict())
    w = list()
    for m in peer.params.models:
        w.append(deepcopy(m.state_dict()))
    # average weights per channel
    for key in wi.keys():
        for wj in w:
            wi[key] += wj[key]
        wi[key] = torch.div(wi[key], len(w) + 1)
    # update calculated weights
    peer.model.load_state_dict(wi)

    return wi


def run_evaluation(graph, history):
    for peer in graph.peers:
        r = peer.model.evaluate(peer.test)
        history[peer.id].append(r)

    return history


def log_round(peer, epoch, history):
    peer_loss = history[peer.id][-1]['val_loss']
    peer_acc = history[peer.id][-1]['val_acc']
    log('', f"Round [{epoch}], {peer}, loss: {peer_loss:.4f}, val_acc: {peer_acc:.4f}")


def old_average_weights(peer, data):
    """
    Returns the average of the weights.
    """
    wi = peer.model.state_dict()
    w = []
    # list of models weights
    for d in data:
        w.append(d['model'].state_dict())
    # average weights per channel
    for key in wi.keys():
        for wj in w:
            wi[key] += wj[key]
        wi[key] = torch.div(wi[key], len(w) + 1)
    # update calculated weights
    peer.model.load_state_dict(wi)

    return peer, wi
