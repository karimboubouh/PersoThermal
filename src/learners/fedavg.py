from time import sleep

from keras.saving.save import load_model
from tqdm import tqdm

from src import protocol
from src.conf import EVAL_ROUND, WAIT_TIMEOUT, WAIT_INTERVAL
from src.helpers import timeit
from src.ml import model_inference, GAR, train_for_x_epochs, train_for_x_batches, meta_train
from src.p2p import Graph, Node
from src.utils import log, wait_until

name = "Federated averaging (FedAvg)"
# Server is peer.neighbors[-1]
NB_BATCHES = 1


def collaborate(graph: Graph, args):
    args.server_id = len(graph.peers) - 1
    server = [peer for peer in graph.peers if peer.id == args.server_id][0]
    log("info", f"Initializing FedAvg...")
    # init peers parameters
    for peer in graph.peers:
        peer.execute(train_init, args)
    graph.join()

    log("info", f"Federated training for T = {graph.args.rounds} rounds")
    T = tqdm(range(graph.args.rounds), position=0)
    for t in T:
        for peer in graph.peers:
            peer.execute(train_step, t, args)
        graph.join(t)

    # stop train
    log("info", f"Evaluating the output of the federated training.")
    for peer in graph.peers:
        peer.execute(train_stop, args)
    graph.join()

    # meta-train
    log("info", f"Meta-Learning using the global federated learned models.")
    FL_MODEL = "FL_MODEL.h5"
    server.model.save(FL_MODEL, save_format="h5")
    peers = [peer for peer in graph.peers if peer.id != args.server_id]
    for peer in peers:
        peer.execute(train_meta, FL_MODEL, args)
        peer.current_exec.join()
    graph.join()
    log('info', f"Graph G disconnected.")

    # get collaboration logs

    collab_logs = {server.id: server.params.logs}
    meta_logs = {peer.id: peer.params.logs for peer in graph.peers if peer.id != server.id}

    return collab_logs, meta_logs


def train_init(peer: Node, args):
    peer.params.exchanges = 0
    if peer.id == args.server_id:
        # server:
        r = peer.evaluate()
        peer.params.logs = [r]
        peer.params.models = {i: [] for i in range(args.rounds)}
    return


def train_step(peer: Node, t, args):
    T = t if isinstance(t, tqdm) or isinstance(t, range) else [t]
    for t in T:
        if peer.id == args.server_id:
            # Server
            wait_until(enough_received, WAIT_TIMEOUT, WAIT_INTERVAL, peer, t, len(peer.neighbors))
            w = GAR(peer, [v for i, v in peer.V[t]])
            update_model(peer, w, evaluate=(t % EVAL_ROUND == 0))
            msg = protocol.train_step(t, peer.get_model_params())  # not grads
            peer.broadcast(msg)
        else:
            if t > 0:
                wait_until(server_received, WAIT_TIMEOUT, WAIT_INTERVAL / 10, peer, t)
                w_server = peer.V[t - 1][0][1]  # [round][n-message(0 in FL)][(id, W)]
                peer.set_model_params(w_server)
            # Worker
            train_for_x_epochs(peer, epochs=peer.params.epochs, evaluate=False, use_tqdm=False)
            # train_for_x_batches(peer, batches=NB_BATCHES, evaluate=False, use_tqdm=False)
            server = peer.neighbors[-1]
            msg = protocol.train_step(t, peer.get_model_params())  # not grads
            peer.send(server, msg)

    return


def update_model(peer: Node, w, evaluate=False):
    peer.set_model_params(w)
    if evaluate:
        t_eval = peer.evaluate(verbose=True)
        peer.params.logs.append(t_eval)


def train_stop(peer: Node, args):
    if peer.id == args.server_id:
        h = model_inference(peer, one_batch=False)
        # history = f"{peer} :: MSE: {h['loss']:4f} | RMSE: {h['rmse']:4f}, MAE: {h['mae']:4f}"
        # peer.broadcast(protocol.log('result', history))
        # only edge will receive if it waits for ~1 secs
        # sleep(1)
        peer.stop()
    # if not meta add
    # peer.stop()


def train_meta(peer: Node, model_file, args):
    train = peer.dataset.generator.train
    peer.model = load_model(model_file)
    r = peer.evaluate()
    log('warning',
        f"Node {peer.id} Fed Model [*] MSE: {r['val_loss']:4f} | RMSE: {r['val_rmse']:4f}, MAE: {r['val_mae']:4f}")
    model, train_history = meta_train(peer.id, model_file, train, epochs=1)
    peer.model = model
    h = model_inference(peer)
    peer.params.logs = [{'FL': r, 'train': train_history, 'test': h}]
    log('', " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    peer.stop()


# ---------- Helper functions -------------------------------------------------

def enough_received(peer: Node, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False


def server_received(peer: Node, t):
    # if t - 1 in peer.V and len(peer.V[t - 1]) == 1:
    # TODO remove after finishing simulations
    if t - 1 in peer.V and len(peer.V[t - 1]) >= 1:
        return True
    return False
