import time

from src import protocol
from src.conf import WAIT_TIMEOUT, WAIT_INTERVAL
from src.ml import train_for_x_batches, train_for_x_epochs
from src.utils import wait_until

name = "Federated averaging (FedAvg)"
# Server is peer.neighbors[-1]
NB_BATCHES = 1


def train_init(peer, args):
    peer.log('warning', f"Learner :: {name}")
    peer.params.exchanges = 0
    return


def train_step(peer, t, args):
    T = t if isinstance(t, range) else [t]
    for t in T:
        if t > 0:
            wait_until(server_received, WAIT_TIMEOUT * 100, WAIT_INTERVAL * 10, peer, t)
            w_server = peer.V[t - 1][0][1]  # [round][n-message(0 in FL)][(id, W)]
            peer.set_model_params(w_server)
        # Worker
        # peer.log('error', f"{peer} :: train_step...", remote=True)
        st = time.perf_counter()
        if args.use_batches:
            train_for_x_batches(peer, batches=NB_BATCHES, evaluate=False, use_tqdm=False)
            et = time.perf_counter() - st
            peer.log('success', f"Round {t} :: took {et:.4f}s to train {NB_BATCHES} batch(es).", remote=False)
        else:
            train_for_x_epochs(peer, epochs=peer.params.epochs, evaluate=False, use_tqdm=False)
            et = time.perf_counter() - st
            peer.log('success', f"Round {t} :: took {et:.4f}s to train {peer.params.epochs} epoch(s).", remote=False)

        msg = protocol.train_step(t, peer.get_model_params())  # not grads
        server = peer.neighbors[-1]
        peer.send(server, msg)

    return


def update_model(peer, w, evaluate=False):
    peer.set_model_params(w)
    if evaluate:
        t_eval = peer.evaluate()
        peer.params.logs.append(t_eval)


def train_stop(peer, args):
    time.sleep(1)
    peer.stop()


# ---------- Helper functions -------------------------------------------------

def enough_received(peer, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False


def server_received(peer, t):
    if t - 1 in peer.V and len(peer.V[t - 1]) == 1:
        return True
    return False
