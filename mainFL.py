import os
import time

import src.conf as C
from src.ecobee import load_p2p_dataset
from src.edge_device import edge_devices
from src.learners import ctm, fedavg
from src.ml import initialize_models
from src.network import central_graph, network_graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, save

if __name__ == '__main__':
    t = time.time()
    args = load_conf(use_cpu=False)
    cluster_id = 0
    args.rounds = 100
    # set rounds dynamic depending on the rmse of CL
    C.TIME_ABSTRACTION = "15min"
    C.RECORD_PER_HOUR = 4
    # Configuration ------------------>
    args.mp = 0  # 1 for mobile phones
    season = 'summer'
    args.model = "LSTM"
    args.learner = "fedavg"
    args.use_batches = True
    args.epochs = 1
    args.batch_size = 128
    resample = False if C.TIME_ABSTRACTION is None else True

    # Details ------------------------>
    fixed_seed(True)
    exp_details(args)
    # Environment setup -------------->

    dataset, input_shape, homes_ids = load_p2p_dataset(args, cluster_id, season, nb_homes=100)  # , nb_homes=10
    models = initialize_models(args.model, input_shape=input_shape, nbr_models=len(dataset), same=True)
    topology = central_graph(models)
    edge = edge_devices(args, count=-1)  # count <= nb_homes-1 to account for the server
    graph = network_graph(topology, models, dataset, homes_ids, args, edge=edge)
    # graph.show_neighbors()
    # graph.show_similarity(ids=True)

    # Federated training ------------->
    train_logs = graph.collaborative_training(learner=fedavg, args=args)

    # Plots -------------------------->
    save(f"FL_{C.ML_ENGINE}_{C.TIME_ABSTRACTION}_cluster_{cluster_id}_{season}_{args.epochs}", train_logs)

    # info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    # plot_train_history(train_logs, metric='rmse', measure="mean-std")
    print(f"END in {round(time.time() - t, 2)}s")
    os._exit(0)
