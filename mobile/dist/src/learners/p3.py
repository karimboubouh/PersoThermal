from heapq import nsmallest

import numpy as np

from src import protocol
from src.conf import EVAL_ROUND, WAIT_TIMEOUT, WAIT_INTERVAL
from src.filters import angular_metric
from src.ml import GAR, model_inference
from src.node import Node
from src.utils import active_peers, wait_until, get_node_conn_by_id

name = "P3 Algorithm"


# ---------- Algorithm functions ----------------------------------------------

def train_init(peer, args):
    peer.log('warning', f"Learner :: {name}")
    peer.log('event', f'Starting collaborative training using {name} ...', remote=False)
    peer.log("info", f"Initializing Collaborative training...", remote=False)
    r = peer.evaluate()
    peer.bridge.send(protocol.call_method("log_results", r))
    peer.params.logs = [r]
    peer.params.exchanges = 0
    peer.params.n_accept = 0
    peer.params.n_reject = 0
    peer.params.delta = 0.8
    peer.params.beta = 1
    peer.params.mu = 0.3
    peer.params.Wi = {n.neighbor_id: 0 for n in peer.neighbors}
    peer.params.E = 5
    peer.params.e = peer.params.E
    peer.params.k = int(np.sqrt(len(peer.neighbors)))
    return


def train_step(peer, t):
    T = t if isinstance(t, range) else [t]
    for t in T:
        # train for E (one) epoch
        peer.train_one_epoch()
        # broadcast current model to all my active neighbors
        active = active_peers(peer.neighbors, peer.params.frac)
        msg = protocol.train_step(t, peer.get_model_params())
        peer.broadcast(msg, active)
        # wait for enough updates labeled with round number t
        wait_until(enough_received, WAIT_TIMEOUT, WAIT_INTERVAL, peer, t, len(active))
        if t not in peer.V:
            peer.V[t] = []
            peer.log('error', f"{peer} received no messages in round {t}.")
        # peer.log('log', f"{peer} got {len(peer.V[t])}/{len(active)} messages in round {t}.", remote=False)
        # collaborativeUpdate
        w_t = collaborativeUpdateLight(peer, t)
        # update and evaluate the model
        # TODO Review update function
        update_model(peer, w_t, evaluate=(t % EVAL_ROUND == 0), t=t)
        # start accepting gradients from next round
        peer.current_round = t + 1
        del peer.V[t]
        # networkUpdate(peer, t, PSS)
    return


def train_stop(peer):
    model_inference(peer, one_batch=True)
    # acceptance_rate = peer.params.n_accept / peer.params.exchanges * 100
    # log('info', f"{peer} Acceptance rate for alpha_max=({peer.params.alpha_max}): {acceptance_rate} %")
    return


def collaborativeUpdate(peer, t):
    w_ref = peer.get_model_params()
    accepted = []
    rejected = []
    for j, w_j in peer.V[t]:
        angle, ED = angular_metric(w_ref.view(1, -1), w_j.view(1, -1), "euclidean")
        # if peer.id == 0 and t % 10 == 0:
        #     log('result', f"{peer}, ref={torch.sum(w_ref)}, w_j={torch.sum(w_j)}, angle={angle}, ED={ED}")
        # if j not in peer.params.Wi:
        #     peer.params.Wi[j] = 0
        # peer.params.Wi[j] = max((1 - peer.params.beta) * peer.params.Wi[j] + peer.params.beta * ED, 0)
        # divergence filter
        if ED <= peer.params.delta or True:
            # todo apply angular filter
            peer.params.n_accept += 1
            accepted.append(w_j)
        else:
            peer.params.n_reject += 1
            rejected.append(w_j)
    if accepted:
        w_gar = peer.params.mu * w_ref + (1 - peer.params.mu) * GAR(peer, accepted)
    else:
        peer.log('log', f"{peer}: No gradients accepted", remote=False)
        w_gar = w_ref

    # update beta
    peer.params.beta = 1 - np.mean([*peer.params.Wi.values()])
    return w_gar


def collaborativeUpdateLight(peer, t):
    w_ref = peer.get_model_params()
    accepted = [w_ref]
    for j, w_j in peer.V[t]:
        accepted.append(w_j)
    w_gar = GAR(peer, accepted)
    return w_gar


def update_model(peer, w_gar, evaluate=False, t=None):
    peer.set_model_params(w_gar)
    # TODO Review update function
    # peer.take_step()
    if evaluate:
        t_eval = peer.evaluate(peer.inference, one_batch=True)
        peer.params.logs.append(t_eval)
        r = f"ROUND[{t}], val_loss: {t_eval['val_loss']:.4f}, val_acc: {t_eval['val_acc']:.4f}"
        peer.log("success", r, remote=False)
        peer.bridge.send(protocol.call_method("log_results", t_eval))


def networkUpdate(peer: Node, t, PSS, tolerate=True):
    if peer.params.e == t:
        if peer.id == 0:
            r = peer.evaluate(peer.inference, one_batch=True)
            peer.log('info', f"{peer} > Round [{t}], val_loss: {r['val_loss']:.4f}, val_acc:  {r['val_acc']:.4f}")
        lowest = nsmallest(peer.params.k, peer.params.Wi, key=peer.params.Wi.get)
        new_neighbors = PSS(peer, len(lowest))  # return List[Node]
        # leave some peers if
        leave = peer.params.k - len(new_neighbors)
        if tolerate and leave > 0:
            del lowest[-leave:]
        # disconnect from neighbors with lowest Wi if they are still connected with the peer
        for low in lowest:
            del peer.params.Wi[low]
            unwanted_peer = get_node_conn_by_id(peer, low)
            if unwanted_peer:
                peer.disconnect(unwanted_peer)
        # connect to new peers
        for new_neighbor in new_neighbors:
            peer.connect(new_neighbor.id, new_neighbor.host, new_neighbor.port)
            peer.params.Wi[new_neighbor.id] = 0
        # set the value of the next illegible round
        peer.params.e = int(peer.params.e + np.sqrt(peer.params.E * t))
        peer.log('log', f"{peer} updated its set of neighbors: {peer.neighbors}", remote=False)
        peer.log('info' if peer.id == 0 else 'log', f"{peer} Next illegible round: {peer.params.e}", remote=False)


# ---------- Helper functions -------------------------------------------------

def enough_received(peer: Node, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False
