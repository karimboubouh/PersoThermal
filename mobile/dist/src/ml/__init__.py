import src.conf as C

if C.ML_ENGINE.lower() == "numpy":
    from src.ml.numpy.models import *
    from src.ml.numpy.helpers import *
elif C.ML_ENGINE.lower() == "n3":
    print("Using N3 ML_ENGINE ...")
    from src.ml.n3.helpers import *
else:
    exit(f'Unknown "{C.ML_ENGINE}" ML engine !')
