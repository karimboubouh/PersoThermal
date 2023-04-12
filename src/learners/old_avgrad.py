import time
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm

from src.conf import RECORD_RATE
from src.p2p import Graph
from src.utils import log, inference_eval, fill_history

name = "Gradient Averaging Collaborative Learner"


def collaborate(graph: Graph, device='cpu'):
    # initialize history holder
    history = dict.fromkeys(range(len(graph.peers)))
    for k in history.keys():
        history[k] = []

    for peer in graph.peers:
        r = peer.model.evaluate(peer.inference, one_batch=True, device=device)
        history[peer.id].append(r)

    # setup algorithm parameters
    for peer in graph.peers:
        peer.params.gradients = {}

    # prepare tqdm
    log("info", f"Collaborative training for {graph.args.rounds} rounds")
    rounds = tqdm(range(graph.args.rounds))
    for epoch in rounds:
        # Randomly activate a peer
        peer = np.random.choice(graph.peers)
        if peer.clustered:
            # exchange with peer's cluster
            peer.train_one_epoch(device)
            for neighbor in peer.neighbors:
                # run one training epoch per neighbor
                neighbor.train_one_epoch(device)
                peer.params.gradients[neighbor.id] = get_update(peer, neighbor)
                neighbor.params.gradients[peer.id] = get_update(neighbor, peer)
            # average grads an take a step
            avg_step(peer, history, device)
        else:
            # Randomly select a neighbor
            neighbor = np.random.choice(peer.neighbors)
            # run one training epoch
            peer.train_one_epoch(device)
            neighbor.train_one_epoch(device)
            # Exchange gradients
            peer.params.gradients[neighbor.id] = get_update(peer, neighbor)
            neighbor.params.gradients[peer.id] = get_update(neighbor, peer)
            # average grads an take a step
            avg_step(peer, history, device)
            avg_step(neighbor, history, device)
            # Show info
            show_info(rounds, peer, neighbor, history)

    # # Evaluate all models every RECORD_RATE
    # if epoch != 0 and epoch % RECORD_RATE == 0:
    #     run_evaluation(graph, history, epoch, device=device)
    #     rounds.set_postfix({**{'peer': peer}, **history[peer.id][-1]})

    log("info", f"Evaluating the output of the collaborative training after {graph.args.rounds} rounds.")
    for peer in graph.peers:
        inference_eval(peer, device)

    return fill_history(history)


def get_update(peer, neighbor):
    Wj = peer.similarity[neighbor.id]
    G = neighbor.get_gradients()
    return (Wj / peer.params.D) * G


def avg_step(peer, history, device):
    # cosine_filter(peer)
    average_grads = average_gradients(peer)
    peer.set_gradients(average_grads, device)
    peer.take_step()
    r = peer.model.evaluate(peer.inference, one_batch=True, device=device)
    history[peer.id].append(r)


def average_gradients(peer):
    grads = [peer.get_gradients()] + list(peer.params.gradients.values())
    return torch.mean(torch.stack(grads), dim=0)


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


def run_evaluation(graph, history, epoch, debug=True, device='cpu'):
    t = time.time()
    current = []
    for peer in graph.peers:
        r = peer.model.evaluate(peer.inference, one_batch=True, device=device)
        history[peer.id].append(r)
        current.append(r)
    if debug:
        current_los = round(np.mean([e['val_loss'] for e in current]), 2)
        current_acc = round(np.mean([e['val_acc'] for e in current]), 2)
        t = round(time.time() - t, 2)
        log('', f"\nEvaluation after {epoch} rounds: mean accuracy: {current_acc} | mean loss {current_los}. ({t}s)\n")

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


def show_info(r, p, n, h, debug=False):
    pl = round(h[p.id][-1]['val_loss'], 3)
    pa = round(h[p.id][-1]['val_acc'], 3)
    nl = round(h[n.id][-1]['val_loss'], 3)
    na = round(h[n.id][-1]['val_acc'], 3)
    p_ = f"{p.id}, loss: {pl}, acc: {pa}"
    n_ = f"{n.id}, loss: {nl}, acc: {na}"
    if debug:
        print()
        log('info', f"Peer: {p_} | Neighbor: {n_}")
    else:
        r.set_postfix({**{'Peer': p_, 'Neighbor': n_}})
