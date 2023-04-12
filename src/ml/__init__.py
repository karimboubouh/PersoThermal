import src.conf as C

if C.ML_ENGINE.lower() == "tensorflow":
    from src.ml.tensorflow.helpers import *
# elif C.ML_ENGINE.lower() == "numpy":
#     from src.ml.numpy.models import *
#     from src.ml.numpy.helpers import *
#     from src.ml.numpy.datasets import get_dataset, train_val_test, inference_ds
# elif C.ML_ENGINE.lower() == "n3":
#     from src.ml.n3.helpers import *
else:
    exit(f'Unknown "{C.ML_ENGINE}" ML engine !')
