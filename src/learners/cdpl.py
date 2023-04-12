import numpy as np
from tqdm import tqdm

from src.conf import RECORD_RATE
from src.p2p import Graph
from src.utils import log

name = "Contribution Driven P2P Learning (CDPL)"


def collaborate(graph: Graph, device='cpu'):
    # initialize history holder
    history = dict.fromkeys(range(len(graph.peers)))
    for k in history.keys():
        history[k] = []

    # setup algorithm parameters
    for peer in graph.peers:
        peer.params.D = sum(peer.similarity.values())
        peer.params.alpha = 0.99
        peer.params.c = calculate_c(peer)
        peer.params.models = {}
        log("info", f"{peer}, D={peer.params.D}, alpha={peer.params.alpha}, C={peer.params.c}")

    # prepare tqdm
    log("info", f"Collaborative training for {graph.args.rounds} rounds")
    rounds = tqdm(range(graph.args.rounds))
    for epoch in rounds:
        # Randomly activate a peer
        peer = np.random.choice(graph.peers)
        # Randomly select a neighbor
        neighbor = np.random.choice(peer.neighbors)
        # Exchange models
        peer.params.models[neighbor.id] = neighbor.model.state_dict()
        neighbor.params.models[peer.id] = peer.model.state_dict()
        # Update models
        update_weights(peer)
        update_weights(neighbor)
        # evaluate all models every RECORD_RATE
        if epoch % RECORD_RATE == 0:
            run_evaluation(graph, history)
            # rounds.set_postfix({**{'peer': peer}, **history[peer.id][-1]})

    for peer in graph.peers:
        print(peer, peer.model.evaluate(peer.test))

    return history


# ----- Custom functions ------------------------------------------------------

def update_weights(peer):
    local_wi = peer.local_model.state_dict()
    wi = peer.model.state_dict()
    # average weights per channel
    for key in wi.keys():
        cj = {k: v[key] for k, v in peer.params.models.items()}
        wi[key] = update_channel(peer, local_wi[key], cj)
    # update calculated weights
    peer.model.load_state_dict(wi)

    return peer, wi


def update_channel(peer, lci, cj):
    sigma = sum([(peer.similarity[k] / peer.params.D) * cj[k] for k in cj])
    c = peer.params.c
    alpha = peer.params.alpha
    alpha_bar = (1 - alpha)
    update = (alpha + alpha_bar * c) ** -1 * (alpha * sigma + alpha_bar * c * lci)

    return update


def run_evaluation(graph, history):
    for peer in graph.peers:
        r = peer.model.evaluate(peer.test)
        history[peer.id].append(r)

    return history


def log_round(peer, epoch, history):
    peer_loss = history[peer.id][-1]['val_loss']
    peer_acc = history[peer.id][-1]['val_acc']
    log('', f"Round [{epoch}], {peer}, loss: {peer_loss:.4f}, val_acc: {peer_acc:.4f}")


def calculate_c(peer):
    peer_data_size = len(peer.train.dataset)
    max_neighbors_data_size = max([len(neighbor.train.dataset) for neighbor in peer.neighbors])
    return peer_data_size / max(peer_data_size, max_neighbors_data_size)
