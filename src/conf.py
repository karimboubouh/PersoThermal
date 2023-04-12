# DATA
DATA_DIR = "data/ecobee/"
DATA_CLUSTERS_DIR = "data/ecobee/clusters"
DATA_SIMILARITY_DIR = "data/ecobee/similarities"
DATA_HOMES_DIR = "data/ecobee/homes"
META_CLUSTER_COLUMNS = ['Identifier', 'Floor Area [ft2]', 'Age of Home [years]']
META_SIMILARITY_COLUMNS = ['Identifier', 'Floor Area [ft2]', 'Number of Floors', 'Age of Home [years]']
DF_CLUSTER_COLUMNS = ['out_hum', 'out_temp', 'mode', 'in_cool', 'in_heat', 'in_hum', 'in_temp']
TIME_ABSTRACTION = "1H"  # None | "H" | 15min
RECORD_PER_HOUR = 1  # 12 for 5min | 4 for 15min | 1 for 1H
LOOK_BACK = 24  # 24 for 1 day | 72 for 3 days | 168 for 1 week
WINTER = ['01', '02']  # '12'
SPRING = ['03', '04', '05']
SUMMER = ['06', '07', '08']
AUTUMN = ['09', '10', '11']

# NETWORK
PORT = 9000
NETWORK_INTERFACE = None
LAUNCHER_PORT = 19491
TCP_SOCKET_BUFFER_SIZE = 5000000
TCP_SOCKET_SERVER_LISTEN = 10
SOCK_TIMEOUT = 20
LAUNCHER_TIMEOUT = 60

# ML change configuration here
# ML_ENGINE = "N3"  # "N3", "TensorFlow", "NumPy", "PyTorch"
ML_ENGINE = "TensorFlow"
DEFAULT_VAL_DS = "val"
DEFAULT_MEASURE = "mean"
EVAL_ROUND = 5
TRAIN_VAL_TEST_RATIO = [.8, .1, .1]
RECORD_RATE = 10
M_CONSTANT = 1
WAIT_TIMEOUT = 600
WAIT_INTERVAL = 0.05  # 0.02
FUNC_TIMEOUT = 600
TEST_SCOPE = 'neighborhood'
IDLE_POWER = 12.60
INFERENCE_BATCH_SIZE = 256
DATASET_DUPLICATE = 0
