import time

import numpy as np

import src.conf as C
from src.ecobee import load_p2p_dataset
from src.edge_device import edge_devices
from src.learners import sp3
from src.ml import initialize_models
from src.network import network_graph, metadata_graph, full_graph
from src.plots import plot_train_history, plot_many_train_history
from src.utils import exp_details, load_conf, fixed_seed, save, get_homes_metadata, load

if __name__ == '__main__':
    # train_logs = load("P4_1H_peers_100_cluster_None_summer_1_0.05-10_B1_794.pkl")
    # for i, j in train_logs.items():
    #     print(f"i={i} -------> {j[-1]}")
    #     print(f"i={i} -------> {j[-2]}")
    #     exit()
    # train_logs = load("P3_30M_Peers_100_R0.05_L10_008C.pkl")
    # train_logs = load("P3_5M_Peers_100_R0.05_L10_020C.pkl")
    # avg_mse = np.mean([tl[-1]["val_loss"] for tl in train_logs.values()], axis=0)
    # std_mse = np.std([tl[-1]["val_loss"] for tl in train_logs.values()], axis=0)
    # avg_rmse = np.mean([tl[-1]["val_rmse"] for tl in train_logs.values()], axis=0)
    # std_rmse = np.std([tl[-1]["val_rmse"] for tl in train_logs.values()], axis=0)
    # a = [tl[-1]['val_rmse'] for tl in train_logs.values()]
    # print([a + avg_rmse])
    # count = sum(x > avg_rmse + std_rmse for x in a)
    # print(f"RMSE : Mean={avg_rmse} | count = {count} our of {len(a)}")
    # avg_mae = np.mean([tl[-1]["val_mae"] for tl in train_logs.values()], axis=0)
    # std_mae = np.std([tl[-1]["val_mae"] for tl in train_logs.values()], axis=0)
    # b = [tl[-1]['val_mae'] for tl in train_logs.values()]
    # countb = sum(x > avg_mae + std_mae for x in b)
    # print(f"MAE : Mean={avg_mae} | count = {countb} our of {len(b)}")
    #
    # r = f"MSE: {avg_mse:4f} (+-{std_mse:4f}); RMSE: {avg_rmse:4f} (+-{std_rmse:4f}) ;MAE: {avg_mae:4f} (+-{std_mae:4f})"
    # print(r)
    # exit()
    # logs = [
    #     {'data': "P3_1H_Peers_100_Cluster_0_006.pkl", 'label': r"Clustering"},
    #     {'data': "P3_1H_Peers_100_R0.01_L1_001.pkl", 'label': r"$\rho=0.01$"},
    #     {'data': "P3_1H_Peers_100_R0.05_L10_002.pkl", 'label': r"$\rho=0.05$"},
    #     {'data': "P3_1H_Peers_100_R0.3_L30_003.pkl", 'label': r"$\rho=0.3$"},
    #     {'data': "P3_1H_Peers_100_R0.6_L60_004.pkl", 'label': r"$\rho=0.6$"},
    #     {'data': "P3_1H_Peers_100_R0.9_L90_005.pkl", 'label': r"$\rho=0.9$"},
    # ]
    # plot_many_train_history(logs, save_fig=True)
    # train_logs = load(f"P3_TensorFlow_30min_peers_100_cluster_None_summer_1_0.05-10_851.pkl")
    # plot_train_history(train_logs, metric='rmse', measure="mean-std", save_fig=True)
    # exit()
    t = time.time()
    # load experiment configuration from CLI arguments
    cpu = False
    args = load_conf(use_cpu=cpu)
    # =================================
    args.mp = 0
    args.model = "LSTM"
    args.learner = "sp3"
    args.batch_size = 512
    args.batches = 1
    args.epochs = 1  # 5
    args.rounds = 300
    cluster_id = None
    season = 'summer'
    C.TIME_ABSTRACTION = "5min"
    C.RECORD_PER_HOUR = 12
    # =================================
    fixed_seed(True)
    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    dataset, input_shape, home_ids = load_p2p_dataset(args, cluster_id, season, nb_homes=100)  # , nb_homes=10
    homes_metadata = get_homes_metadata(home_ids, normalize=False)
    # build users models
    models = initialize_models(args.model, input_shape=input_shape, nbr_models=len(dataset), same=True)
    # generate network topology
    # topology = full_graph(models)
    rho = 0.05
    limit = 10
    topology = metadata_graph(homes_metadata, dist="euclidian", rho=rho, limit=limit)

    # include physical edge devices  (count < 1 to only use simulated nodes)
    edge = edge_devices(args, count=-1)
    # build the network graph
    graph = network_graph(topology, models, dataset, home_ids, args, edge=edge)
    graph.show_neighbors()
    # exit()
    # graph.show_similarity(ids=False)
    # Phase I: Local Training

    graph.local_training(one_batch=False, inference=True)

    # Phase II: Collaborative training
    train_logs = graph.collaborative_training(learner=sp3, args=args)
    save(
        f"P4_{C.TIME_ABSTRACTION}_peers_{args.num_users}_cluster_{cluster_id}_{season}_{args.epochs}_{rho}-{limit}_B{args.batches}",
        train_logs)
    avg_mse = np.mean([tl[-1]["loss"] for tl in train_logs.values()], axis=0)
    std_mse = np.std([tl[-1]["loss"] for tl in train_logs.values()], axis=0)
    avg_rmse = np.mean([tl[-1]["rmse"] for tl in train_logs.values()], axis=0)
    std_rmse = np.std([tl[-1]["rmse"] for tl in train_logs.values()], axis=0)
    avg_mae = np.mean([tl[-1]["mae"] for tl in train_logs.values()], axis=0)
    std_mae = np.std([tl[-1]["mae"] for tl in train_logs.values()], axis=0)
    r = f"MSE: {avg_mse:4f} (+-{std_mse:4f}); RMSE: {avg_rmse:4f} (+-{std_rmse:4f}) ;MAE: {avg_mae:4f} (+-{std_mae:4f})"
    print(r)
    print(f"args.model = {args.model} | args.batch_size = {args.batch_size} | C.TIME_ABSTRACTION={C.TIME_ABSTRACTION}")
    # plot_train_history(train_logs, metric='rmse', measure="mean-std", save_fig=True)
    print(f"END in {round(time.time() - t, 2)}s")

"""  
    # load plots
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    train_logs = load("p3_log_100_10_880.pkl")
    plot_train_history(train_logs, metric='rmse', measure="mean")
    exit()
"""
