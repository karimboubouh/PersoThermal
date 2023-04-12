import numpy as np

import src.conf as C
from src.ecobee import load_p2p_dataset
from src.ml import initialize_models
from src.network import network_graph, disconnected_graph
from src.utils import exp_details, load_conf, fixed_seed, save, log, load

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf(use_cpu=False)
    # =================================
    args.model = "LSTM"
    args.batch_size = 128
    args.epochs = 5
    cluster_id = 0
    season = 'summer'
    C.TIME_ABSTRACTION = "1H"
    C.RECORD_PER_HOUR = 1
    resample = False if C.TIME_ABSTRACTION is None else True
    # =================================
    fixed_seed(True)
    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    dataset, input_shape, homes_ids = load_p2p_dataset(args, cluster_id, season, resample=resample, nb_homes=100)

    # build users models
    models = initialize_models(args.model, input_shape=input_shape, nbr_models=len(dataset), same=True)
    topology = disconnected_graph(models)
    # build the network graph
    graph = network_graph(topology, models, dataset, homes_ids, args)

    # Local Training
    train_logs = graph.local_training(one_batch=False, inference=True)
    save(f"LL_{C.ML_ENGINE}_{C.TIME_ABSTRACTION}_cluster_{cluster_id}_{season}_{args.epochs}", train_logs)
    avg_mse = np.mean([tl['test']['loss'] for tl in train_logs.values()], axis=0)
    std_mse = np.std([tl['test']['loss'] for tl in train_logs.values()], axis=0)
    avg_rmse = np.mean([tl['test']['rmse'] for tl in train_logs.values()], axis=0)
    std_rmse = np.std([tl['test']['rmse'] for tl in train_logs.values()], axis=0)
    avg_mae = np.mean([tl['test']['mae'] for tl in train_logs.values()], axis=0)
    std_mae = np.std([tl['test']['mae'] for tl in train_logs.values()], axis=0)
    r = f"MSE: {avg_mse:4f} (+-{std_mse:4f}); RMSE: {avg_rmse:4f} (+-{std_rmse:4f}) ;MAE: {avg_mae:4f} (+-{std_mae:4f})"
    log('warning', f"AVG Inference for Local Learning >> {r}")

    print("END.")
