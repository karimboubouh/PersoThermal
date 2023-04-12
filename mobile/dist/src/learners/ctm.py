import numpy as np

from src import protocol
from src.conf import EVAL_ROUND, WAIT_TIMEOUT, WAIT_INTERVAL, ML_ENGINE
from src.ml import GAR, model_inference
from src.utils import active_peers, wait_until, norm_squared

name = "Static Personalized P2P (SP3)"


# ---------- Algorithm functions ----------------------------------------------

def train_init(peer):
    r = peer.evaluate()
    peer.params.logs = [r]
    peer.params.ar = 0  # acceptance_rate
    peer.params.exchanges = 0
    peer.params.n_accept = 0
    peer.params.n_reject = 0
    peer.params.sigma = 10
    peer.params.D = []
    peer.params.mu = 0.5
    peer.params.Wi = {n.neighbor_id: 0 for n in peer.neighbors}
    return


def train_step(peer, t):
    T = t if isinstance(t, range) else [t]
    for t in T:
        # train for E (one) epoch
        peer.train_one_epoch()  # weights ==> multiple epochs
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
            peer.log('error', f"{peer} received no messages in round {t}.")
        else:
            peer.log('log', f"{peer} got {len(peer.V[t])}/{len(active)} messages in round {t}.")
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
    model_inference(peer)
    acceptance_rate = round(peer.params.n_accept / peer.params.exchanges * 100, 2)
    peer.params.ar = acceptance_rate
    peer.stop()
    return


def collaborativeUpdate(peer, t):
    vi: list = peer.get_model_params()
    accepted = []
    rejected = []
    # Similarity filter
    for j, vj in peer.V[t]:
        diff = norm_squared(vi, vj)
        # TODO Evaluate each layer and compare per layer vi and vj
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
        peer.log('log', f"{peer}: No accepted gradients in round {t}")
        return vi


def update_model(peer, v, evaluate=False):
    peer.set_model_params(v)
    # TODO Review update function
    # peer.take_step()
    if evaluate:
        t_eval = peer.evaluate()
        peer.params.logs.append(t_eval)


# ---------- Helper functions -------------------------------------------------

def enough_received(peer, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False


def estimate_sigma(peer):
    if peer.params.sigma is None:
        sigma = np.median([norm_squared(peer.get_model_params(), vj) for _, vj in peer.V[0]])
        peer.params.sigma = round(2 * float(sigma), 2)
        peer.log('info', f"{peer} estimated sigma: {peer.params.sigma}")
    return
