import time
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from heapq import nsmallest
import traceback

from src import protocol
from src.aggregators import GAR
from src.filters import angular_metric
from src.p2p import Graph, Node
from src.utils import log, inference_eval, active_peers, wait_until, get_node_conn_by_id

name = "Gradient Averaging Collaborative Learner"


# ---------- MAlgorithm functions ----------------------------------------------

def collaborate(graph: Graph, args):
    log("info", f"Initializing Collaborative training...")
    # init peers parameters
    for peer in graph.peers:
        peer.execute(train_init)
    graph.join()

    log("info", f"Collaborative training for T = {graph.args.rounds} rounds")
    # run the algorithm for T rounds
    # T = tqdm(range(graph.args.rounds), position=0)
    # r = time.time()
    # for t in T:
    #     for peer in graph.peers:
    #         peer.execute(train_step, t, graph.PSS)
    #     graph.join(t)
    # r = time.time() - r
    # log("info", f"train_step took {r:.2f} seconds.")

    for peer in graph.peers:
        T = tqdm(range(graph.args.rounds), position=0, desc=f"{peer}")
        peer.execute(train_step, T, graph.PSS)
    graph.join()

    # stop train
    log("info", f"Evaluating the output of the collaborative training.")
    for peer in graph.peers:
        peer.execute(train_stop)
    graph.join()
    log('info', f"Graph G disconnected.")

    # get collaboration logs
    collab_logs = {}
    for peer in graph.peers:
        collab_logs[peer.id] = peer.params.logs
        peer.join()

    return collab_logs


# ---------- Algorithm functions ----------------------------------------------

def train_init(peer):
    r = peer.model.evaluate(peer.inference, one_batch=True, device=peer.device)
    peer.params.logs = [r]
    peer.params.exchanges = 0
    peer.params.n_accept = 0
    peer.params.n_reject = 0
    peer.params.alpha_max = 360.0
    peer.params.beta = 1
    peer.params.mu = 0.5
    peer.params.Wi = {n.neighbor_id: 0 for n in peer.neighbors}
    peer.params.E = 5
    peer.params.e = peer.params.E
    peer.params.k = int(np.sqrt(len(peer.neighbors)))
    return


def train_step(peer, t, PSS):
    T = t if isinstance(t, tqdm) or isinstance(t, range) else [t]
    for t in T:
        # train one epoch
        peer.train_one_epoch()
        # broadcast gradients
        active = active_peers(peer.neighbors, peer.params.frac)
        msg = protocol.train_step(t, peer.get_gradients())
        peer.broadcast(msg, active)
        # wait for enough updates labeled with round number t
        wait_until(enough_grads, 3, 0.05, peer, t, len(active))
        if t not in peer.V:
            peer.V[t] = []
            log('error', f"{peer} received no messages in round {t}")
        else:
            log('log', f"{peer} -- T= {t} -- Got enough messages : {len(peer.V[t])}.")
        # collaborativeUpdate
        v_t = collaborativeUpdateLight(peer, t)
        # update and evaluate the model
        # TODO Review update function
        update_model(peer, v_t, evaluate=(t % 10 == 0))
        # start accepting gradients from next round
        peer.current_round = t + 1
        del peer.V[t]
        # networkUpdate(peer, t, PSS)
    return


def train_stop(peer):
    inference_eval(peer)
    acceptance_rate = peer.params.n_accept / peer.params.exchanges * 100
    log('info', f"{peer} Acceptance rate for alpha_max=({peer.params.alpha_max}): {acceptance_rate} %")
    peer.stop()
    return


def collaborativeUpdate(peer, t):
    v_ref = peer.get_gradients()
    accepted = []
    rejected = []
    for j, v_j in peer.V[t]:
        # todo discuss: use norm squared, for Wij find a different formula to update
        alpha, gamma = angular_metric(v_ref.view(1, -1), v_j.view(1, -1), "cosine", similarity=True)
        # print(f"{peer}, ({j}) | alpha: {alpha} | gamma:{gamma} max: {peer.params.alpha_max}| "
        #       f"Wij: {len(peer.params.Wi)} | beta: {peer.params.beta}")
        if j not in peer.params.Wi:
            peer.params.Wi[j] = 0
        peer.params.Wi[j] = max((1 - peer.params.beta) * peer.params.Wi[j] + peer.params.beta * gamma, 0)
        if alpha <= peer.params.alpha_max:
            peer.params.n_accept += 1
            accepted.append(v_j)
        else:
            peer.params.n_reject += 1
            rejected.append(v_j)
    if accepted:
        v_gar = peer.params.mu * v_ref + (1 - peer.params.mu) * GAR(peer, accepted)
    else:
        log('log', f"{peer}: No gradients accepted")
        v_gar = v_ref

    # update beta
    peer.params.beta = 1 - np.mean([*peer.params.Wi.values()])
    return v_gar


def collaborativeUpdateLight(peer, t):
    v_ref = peer.get_gradients()
    accepted = [v_ref]
    for j, v_j in peer.V[t]:
        accepted.append(v_j)
    v_gar = GAR(peer, accepted)
    return v_gar


def update_model(peer, grad, evaluate=True):
    peer.set_gradients(grad)
    # TODO Review update function
    peer.take_step()
    if evaluate:
        t_eval = peer.model.evaluate(peer.inference, one_batch=True, device=peer.device)
        peer.params.logs.append(t_eval)


def networkUpdate(peer: Node, t, PSS, tolerate=True):
    if peer.params.e == t:
        if peer.id == 0:
            r = peer.model.evaluate(peer.inference, peer.device, one_batch=True)
            log('info', f"{peer} > Round [{t}], val_loss: {r['val_loss']:.4f}, val_acc:  {r['val_acc']:.4f}")
        lowest = nsmallest(peer.params.k, peer.params.Wi, key=peer.params.Wi.get)
        new_neighbors = PSS(peer, len(lowest))  # return List[Node]
        # leave some peers if
        leave = peer.params.k - len(new_neighbors)
        if tolerate and leave > 0:
            del lowest[-leave:]
        # disconnect from neighbors with lowest Wi if they are still connected with the peer
        for l in lowest:
            del peer.params.Wi[l]
            unwanted_peer = get_node_conn_by_id(peer, l)
            if unwanted_peer:
                peer.disconnect(unwanted_peer)
        # connect to new peers
        for new_neighbor in new_neighbors:
            peer.connect(new_neighbor)
            peer.params.Wi[new_neighbor.id] = 0
        # set the value of the next illegible round
        peer.params.e = int(peer.params.e + np.sqrt(peer.params.E * t))
        log('log', f"{peer} updated its set of neighbors: {peer.neighbors}")
        log('info' if peer.id == 0 else 'log', f"{peer} Next illegible round: {peer.params.e}")


# ---------- Helper functions -------------------------------------------------

def enough_grads(peer: Node, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False


def get_update(peer, neighbor):
    Wj = peer.similarity[neighbor.id]
    G = neighbor.get_gradients()
    return (Wj / peer.params.D) * G


def avg_step(peer, grad, history, device):
    peer.set_gradients(grad, device)
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
        current_los = 0  # round(np.mean([e['val_loss'] for e in current]), 2)
        current_acc = 0  # round(np.mean([e['val_acc'] for e in current]), 2)
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
