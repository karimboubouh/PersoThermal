import time

import src.conf as C
from src import plots
from src.ecobee import read_ecobee_cluster, prepare_ecobee, make_predictions
from src.helpers import Map
from src.ml import initialize_models
from src.p2p import Node
from src.utils import load_conf, fixed_seed, exp_details

if __name__ == '__main__':
    t = time.time()
    # Configuration ------------------>
    args = load_conf(use_cpu=False)
    cluster_id = 0
    season = 'summer'
    args.model = "LSTM"
    args.epochs = 5
    args.batch_size = 128
    C.TIME_ABSTRACTION = "15min"
    C.RECORD_PER_HOUR = 4
    resample = False if C.TIME_ABSTRACTION is None else True
    # Details ------------------------>
    fixed_seed(True)
    exp_details(args)
    # Load data ---------------------->
    data = read_ecobee_cluster(cluster_id, season, resample=resample)
    n_input = C.LOOK_BACK * C.RECORD_PER_HOUR
    n_features = len(C.DF_CLUSTER_COLUMNS)
    shape = (n_input, n_features)
    ds, info = prepare_ecobee(data[season], season, ts_input=n_input, batch_size=args.batch_size)
    # Initialize model --------------->
    model = initialize_models(args.model, input_shape=shape, nbr_models=1, same=True)[0]
    # Centralized Training ----------->
    server = Node(0, model, ds, [], False, {}, args)
    history = server.fit(args, one_batch=False, inference=True)
    predictions = make_predictions(server, info, ptype="test")
    server.stop()
    print(f"END in {round(time.time() - t, 2)}s")
    # Plots -------------------------->
    info = Map({'xlabel': "Time period", 'ylabel': 'Temperature'})
    plots.plot_predictions(predictions, info)

