from time import sleep

import numpy as np
from tqdm import tqdm

from src import protocol
from src.conf import EVAL_ROUND, WAIT_TIMEOUT, WAIT_INTERVAL, ML_ENGINE
from src.ml import GAR, model_inference
from src.p2p import Graph, Node
from src.utils import log, active_peers, wait_until, norm_squared

name = "Static Personalized P2P (SP3)"


# ---------- Algorithm functions ----------------------------------------------

def collaborate(graph: Graph, args):
    log("info", f"Initializing Collaborative training...")
    # init peers parameters
    for peer in graph.peers:
        peer.execute(train_init, args)
    graph.join()

    log("info", f"Collaborative training for T = {graph.args.rounds} rounds")
    T = tqdm(range(graph.args.rounds), position=0)
    for t in T:
        for peer in graph.peers:
            peer.execute(train_step, t)
        graph.join(t)

    # stop train
    log("info", f"Evaluating the output of the collaborative training.")
    for peer in graph.peers:
        peer.execute(train_stop)
    graph.join()
    log('info', f"Graph G disconnected.")

    # get collaboration logs
    collab_logs = {peer.id: peer.params.logs for peer in graph.peers}
    return collab_logs


# ---------- Algorithm functions ----------------------------------------------

def train_init(peer: Node, args):
    r = peer.evaluate()
    peer.params.logs = [r]
    peer.params.exchanges = 0
    peer.params.n_accept = 0
    peer.params.n_reject = 0
    peer.params.sigma = 10
    peer.params.D = []
    peer.params.mu = 0.5
    peer.params.Wi = {n.neighbor_id: 0 for n in peer.neighbors}
    return


def train_step(peer, t):
    T = t if isinstance(t, tqdm) or isinstance(t, range) else [t]
    for t in T:
        # train for E (one) epoch
        peer.train_one_epoch(batches=peer.params.batches)  # weights ==> multiple epochs
        # train_for_x_epoch(peer, 10)
        # broadcast current model to all my active neighbors
        active = active_peers(peer.neighbors, peer.params.frac)
        # TODO check exchanging grads instead of model params.
        msg = protocol.train_step(t, peer.get_model_params())
        peer.broadcast(msg, active)
        # wait for enough updates labeled with round number t
        wait_until(enough_received, WAIT_TIMEOUT, WAIT_INTERVAL, peer, t, len(active))
        if t not in peer.V:
            peer.V[t] = []
            log('error', f"{peer} received no messages in round {t}.")
        # estimate \sigma in first round
        estimate_sigma(peer)
        # collaborativeUpdate
        v = collaborativeUpdate(peer, t)
        # update and evaluate the model
        update_model(peer, v, evaluate=(t % EVAL_ROUND == 0))
        # start accepting gradients from next round
        peer.current_round = t + 1
        del peer.V[t]
    return


def train_stop(peer):
    h = model_inference(peer, one_batch=False)
    peer.params.logs.append(h)
    log('info', f"{peer} disconnecting...")
    sleep(1)
    peer.stop()
    return


def collaborativeUpdate(peer, t):
    vi: list = peer.get_model_params()
    accepted = []
    rejected = []
    # Similarity filter
    for j, vj in peer.V[t]:
        diff = norm_squared(vi, vj)
        peer.params.D.append(diff)
        if diff < peer.params.sigma:
            peer.params.n_accept += 1
            accepted.append(vj)
        else:
            peer.params.n_reject += 1
            rejected.append(vj)
    # Aggregation using personalization parameter mu
    if accepted:
        if ML_ENGINE == "PyTorch":
            return peer.params.mu * vi + (1 - peer.params.mu) * GAR(peer, accepted)
        else:
            avg = GAR(peer, accepted)
            return [peer.params.mu * vi_k + (1 - peer.params.mu) * avg[k] for k, vi_k in enumerate(vi)]
    else:
        log('log', f"{peer}: No accepted gradients in round {t}")
        return vi


def update_model(peer, v, evaluate=False):
    peer.set_model_params(v)
    # TODO Review update function
    # peer.take_step()
    if evaluate:
        t_eval = peer.evaluate(one_batch=True)
        peer.params.logs.append(t_eval)


# ---------- Helper functions -------------------------------------------------

def enough_received(peer: Node, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False


def estimate_sigma(peer: Node):
    if peer.params.sigma is None:
        sigma = np.median([norm_squared(peer.get_model_params(), vj) for _, vj in peer.V[0]])
        peer.params.sigma = round(2 * float(sigma), 2)
        log('info', f"{peer} estimated sigma: {peer.params.sigma}")
    return
