import time
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import src.conf as C
from src import plots
from src.helpers import Map
from src.p2p import Graph
from src.plots import box_plot, scatter_plot, boxs
from src.utils import load_conf, fixed_seed, exp_details, save, load

if __name__ == '__main__':
    t = time.time()
    args = load_conf(use_cpu=True)
    # Configuration ------------------>
    cluster_id = 0  # 0    5484672
    season = 'summer'  # 'summer'
    args.model = "LSTM"
    args.epochs = 5  # 5
    args.batch_size = 128
    C.TIME_ABSTRACTION = "15min"
    C.RECORD_PER_HOUR = 4
    resample = False if C.TIME_ABSTRACTION is None else True

    # Details ------------------------>
    fixed_seed(True)
    exp_details(args)

    # Centralized Training ----------->
    results = Graph.centralized_training(args, cluster_id, season, resample, predict=True, hval=True, meta=True)
    save(f"CL_{C.ML_ENGINE}_{C.TIME_ABSTRACTION}_cluster_{cluster_id}_{season}_{args.epochs}", results)
    train_log, predictions, n_steps_predictions, homes_logs, meta_logs = results
    ""
    # Plots -------------------------->
    # plot predictions
    info = Map({'xlabel': "Time period", 'ylabel': 'Temperature'})
    plots.plot_predictions(predictions, info)
    plots.plot_predictions(n_steps_predictions, info)
    # plot box_plot
    box_plot(train_log, homes_logs, showfliers=True)
    # plot scatter_plot
    scatter_plot(homes_logs, meta_logs)

    # END ---------------------------->
    print(f"END in {round(time.time() - t, 2)}s")
