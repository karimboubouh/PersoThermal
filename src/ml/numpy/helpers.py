import copy
import time

import numpy as np
import tensorflow as tf

from .models import DNN_ARCH, RNN_ARCH, LSTM_ARCH
from .Model import Model
from src.utils import log
from .aggregators import average, median, aksel, krum
from .utils import flatten_grads, unflatten_grad
from ...helpers import Map


def initialize_models(model_name, input_shape, nbr_models=1, same=False):
    models = []
    if same:
        # Initialize all models with same weights
        model = build_model(model_name, input_shape)
        if nbr_models == 1:
            models.append(model)
        else:
            for i in range(nbr_models):
                models.append(copy.deepcopy(model))
    else:
        # Independent initialization
        for i in range(nbr_models):
            models.append(build_model(model_name, input_shape))

    return models


def build_model(model_name, input_shape):
    model = None
    if model_name == 'RNN':
        model = Model(name=model_name, input_dim=input_shape).initial(RNN_ARCH)
    elif model_name == 'LSTM':
        model = Model(name=model_name, input_dim=input_shape).initial(LSTM_ARCH)
    elif model_name == 'DNN':
        model = Model(name=model_name, input_dim=input_shape).initial(DNN_ARCH)
    elif model_name == 'BNN':
        raise NotImplementedError()
    else:
        exit('Error: Unrecognized model')

    return model


def model_fit(peer, tqdm_bar=False):
    peer.model.train(peer.dataset.generator.train.X, peer.dataset.generator.train.Y)
    peer.model.val(peer.dataset.generator.test.X, peer.dataset.generator.test.Y)
    peer.model.test(peer.dataset.generator.test.X, peer.dataset.generator.test.Y)
    history = peer.model.fit(
        lr=peer.params.lr,
        momentum=peer.params.momentum,
        max_epoch=peer.params.epochs,
        batch_size=peer.params.batch_size,
        evaluation=True,
        logger=log
    )

    _Y = peer.model.measure(peer.dataset.generator.test.X, peer.dataset.generator.test.Y)
    return history


def train_for_x_batches(peer, batches=1, evaluate=False):
    # TODO improve FedAvg for numpy
    if peer.model.has_no_data():
        peer.model.train(peer.train.dataset, peer.train.targets)
        peer.model.val(peer.val.dataset, peer.val.targets)
        peer.model.test(peer.test.dataset, peer.test.targets)
    return peer.model.improve(batches, evaluate)


def evaluate_model(model, dataholder, one_batch=False, device=None):
    loss, acc = model.evaluate(dataholder.dataset, dataholder.targets, one_batch=one_batch)
    return {'val_loss': loss, 'val_acc': acc}


def model_inference(peer, batch_size=16, one_batch=False):
    t = time.time()
    dataset, targets = peer.dataset.generator.test.X, peer.dataset.generator.test.Y
    loss, acc = peer.model.evaluate(dataset, targets, one_batch)
    o = "1B" if one_batch else "*B"
    t = time.time() - t
    log('result', f"{peer} [{t:.2f}s] {o} Inference loss: {loss:.4f},  acc: {(acc * 100):.2f}%")


def evaluate_home(home_id, model, generator, batch_size=16, one_batch=False, dtype="Test "):
    if one_batch:
        batch = np.random.choice(len(generator), 1)
        X, y = generator[batch]
        h = model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
    else:
        h = model.evaluate(generator, verbose=0, batch_size=batch_size)
    one = "[^]" if one_batch else "[*]"
    history = Map({'loss': h[0], 'rmse': h[1], 'mae': h[2]})
    log('result', f"Home {home_id} || {dtype} {one} MSE: {h[0]:4f} | RMSE: {h[1]:4f} | MAE {h[2]:4f}")
    return history


def get_params(model, named=False, numpy=None):
    if named:
        return model.named_parameters()
    else:
        return model.parameters


def set_params(model, params, named=False, numpy=None):
    if named:
        log("error", "Setting params using named params is not supported")
        exit()
    else:
        model.parameters = params


def GAR(peer, grads, weighted=True):
    # Weighted Gradients Aggregation rule
    flattened = flatten_grads(grads)
    if peer.params.gar == "average":
        r = average(flattened)
    elif peer.params.gar == "median":
        r = median(flattened)
    elif peer.params.gar == "aksel":
        r = aksel(flattened)
    elif peer.params.gar == "krum":
        r = krum(flattened)
    else:
        raise NotImplementedError()
    return unflatten_grad(r, grads[0])


def timeseries_generator(X_train, X_test, Y_train, Y_test, length, batch_size=None):
    # TODO Review this line
    batch_size = None

    train_x, train_y, train_generator = data2timeseries(X_train, Y_train, length, batch_size=batch_size)
    test_x, test_y, test_generator = data2timeseries(X_test, Y_test, length, batch_size=batch_size)

    if isinstance(batch_size, int):
        return train_generator, test_generator
    else:
        return Map({'X': train_x, 'Y': train_y}), Map({'X': test_x, 'Y': test_y})


def data2timeseries(xval, yval, length, batch_size=None):
    _X = []
    _Y = []
    for i in range(length, xval.shape[0]):
        _X.append(xval[i - length:i])
        _Y.append(yval[i][-1])
    _X = np.array(_X)
    _Y = np.array(_Y).reshape(-1, 1)

    if isinstance(batch_size, int):
        batches = []
        for i in range(0, _X.shape[0], batch_size):
            _Xb = _X[i:min(i + batch_size, _X.shape[0]), :]
            _Yb = _Y[i:min(i + batch_size, _X.shape[0]), :]
            batches.append([_Xb, _Yb])
    else:
        batches = None

    return _X, _Y, batches
