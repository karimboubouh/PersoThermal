import copy
import os
import time

import numpy as np
from .shinnosuke import layers as L, StochasticGradientDescent, Adam
from .shinnosuke.models import Sequential, Model

from tqdm import tqdm
from .aggregators import average, median, aksel, krum
from ...helpers import Map, timeit
from ...utils import log
from tqdm.keras import TqdmCallback


def initialize_models(model_name, input_shape, n_outputs=1, cpu=False, nbr_models=1, same=False):
    models = []
    if same:
        # Initialize all models with same weights
        model, custom_metrics = build_model(model_name, input_shape, n_outputs)
        if nbr_models == 1:
            models.append(model)
        else:
            for i in range(nbr_models):
                models.append(copy.deepcopy(model))
    else:
        # Independent initialization
        for i in range(nbr_models):
            model, _ = build_model(model_name, input_shape, cpu)
            models.append(model)

    return models


def build_model(model_name, input_shape: tuple, n_outputs=1):
    input_shape = (None,) + input_shape
    model = Sequential()
    if model_name == 'RNN':
        model.add(L.SimpleRNN(100, activation='relu', input_shape=input_shape))
    elif model_name == 'LSTM':
        model.add(L.LSTM(100, activation='tanh', input_shape=input_shape))
    elif model_name == 'DNN':
        model.add(L.Dense(100, activation='tanh', input_shape=(input_shape[1],)))
    elif model_name == 'BNN':
        raise NotImplementedError()
    else:
        exit('Error: Unrecognized model')
    # model.add(L.Dropout(0.2))
    model.add(L.Dense(n_outputs))
    # sgd = StochasticGradientDescent(lr=0.001)
    sgd = StochasticGradientDescent(lr=0.1)
    model.compile(optimizer=sgd, loss='mse')
    return model, None


def model_fit(peer, tqdm_bar=False):
    # Map({'X': trainX, 'Y': trainY}), Map({'X': testX, 'Y': testY})
    train = peer.dataset.generator.train
    test = peer.dataset.generator.test
    X, Y = train['X'], train['Y']
    val = (test['X'], test['Y'])
    history = peer.model.fit(X, Y, shuffle=False, epochs=peer.params.epochs, validation_data=val,
                             batch_size=peer.params.batch_size, draw_acc_loss=False, verbose=True, log=print)
    h = list(history.values())
    log('result',
        f"Node {peer.id} Train MSE: {h[0][-1]:4f}, RMSE: {h[1][-1]:4f} | MAE {h[2][-1]:4f}")
    return history


def meta_train(i, model_file, train, batch_size, epochs=1):
    log('event', f"Home {i} performs personalized learning using local data for {epochs} epochs...")
    model = load_model(model_file)
    model.fit(train, epochs=epochs, batch_size=batch_size)
    history = model.history.history
    h = list(history.values())
    log('success', f"Node {i} META Train MSE: {h[0][-1]:4f}, RMSE: {h[1][-1]:4f} | MAE {h[2][-1]:4f}")

    return model, history


def train_for_x_epochs(peer, epochs=1, evaluate=False, use_tqdm=None, verbose=False):
    h1 = Map({'loss': [], 'rmse': [], 'mae': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []})
    train = peer.dataset.generator.train
    test = peer.dataset.generator.test
    bsize = peer.params.batch_size
    X, Y = train['X'], train['Y']
    val = (test['X'], test['Y'])
    if evaluate:
        h = peer.model.fit(X, Y, validation_data=val, shuffle=False, epochs=epochs, batch_size=bsize, verbose=verbose)
        h1.loss.append(h['loss'])
        h1.rmse.append(h['rmse'])
        h1.mae.append(h['mae'])
        h1.val_loss.append(h['val_loss'])
        h1.val_rmse.append(h['val_rmse'])
        h1.val_mae.append(h['val_mae'])
    else:
        # t = time.time()
        h = peer.model.fit(X, Y, validation_ratio=0, shuffle=False, epochs=epochs, batch_size=bsize, verbose=verbose)
        # print(f"{peer} training for {epochs} epochs took {(time.time() - t):5f} seconds.")
        h1.loss.append(h['loss'])
        h1.rmse.append(h['rmse'])
        h1.mae.append(h['mae'])

    return h1


def train_for_x_batches(peer, batches=1, evaluate=False, use_tqdm=True, verbose=False):
    h1 = Map({'loss': [], 'rmse': [], 'mae': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []})
    T = tqdm(range(batches), position=0) if use_tqdm else range(batches)
    train = peer.dataset.generator.train
    test = peer.dataset.generator.test
    X, Y = train['X'], train['Y']
    val = (test['X'], test['Y'])
    batch_size = peer.params.batch_size
    nb_batches = len(Y) // batch_size
    for _ in T:
        i = np.random.choice(range(nb_batches), 1).item()
        bX, bY = X[batch_size * i:batch_size * (i + 1)], Y[batch_size * i:batch_size * (i + 1)]
        bsize = len(bY)
        if evaluate:
            h = peer.model.fit(bX, bY, validation_data=val, shuffle=False, epochs=1, batch_size=bsize, verbose=verbose)
            h1.loss.append(h['loss'])
            h1.rmse.append(h['rmse'])
            h1.mae.append(h['mae'])
            h1.val_loss.append(h['val_loss'])
            h1.val_rmse.append(h['val_rmse'])
            h1.val_mae.append(h['val_mae'])
        else:
            h = peer.model.fit(bX, bY, validation_ratio=0, shuffle=False, epochs=1, batch_size=bsize, verbose=verbose)
            h1.loss.append(h['loss'])
            h1.rmse.append(h['rmse'])
            h1.mae.append(h['mae'])

    return h1


def model_inference(peer, batch_size=16, one_batch=False):
    test = peer.dataset.generator.test
    X, Y = test['X'], test['Y']
    if one_batch:
        nb_batches = len(Y) // batch_size
        # mini_batch_X = shuffled_X[batch_size * i:batch_size * (i + 1)]
        i = np.random.choice(range(nb_batches), 1).item()
        bX, bY = X[batch_size * i:batch_size * (i + 1)], Y[batch_size * i:batch_size * (i + 1)]
        h = peer.model.evaluate(bX, bY, batch_size=None)
    else:
        h = peer.model.evaluate(X, Y, batch_size=None)
    # rmse, mae, mse
    history = Map({'loss': h[0], 'rmse': h[1], 'mae': h[2]})
    one = "[^]" if one_batch else "[*]"
    log('result', f"Node {peer.id} Inference {one} MSE: {h[0]:4f} | RMSE: {h[1]:4f}, MAE: {h[2]:4f}")
    return history


def evaluate_model(peer, one_batch=False, batch_size=64, verbose=False):
    test = peer.dataset.generator.test
    X, Y = test['X'], test['Y']
    if one_batch:
        nb_batches = len(Y) // batch_size
        # mini_batch_X = shuffled_X[batch_size * i:batch_size * (i + 1)]
        i = np.random.choice(range(nb_batches), 1).item()
        bX, bY = X[batch_size * i:batch_size * (i + 1)], Y[batch_size * i:batch_size * (i + 1)]
        h = peer.model.evaluate(bX, bY, batch_size=None)
    else:
        h = peer.model.evaluate(X, Y, batch_size=None)

    if verbose:
        log('result', f"MSE: {h[0]:4f} | RMSE: {h[1]:4f}, MAE: {h[2]:4f}")

    return {'val_loss': h[0], 'val_rmse': h[1], 'val_mae': h[2]}


def evaluate_home(home_id, model, generator, batch_size=16, one_batch=False, dtype="Test "):
    if one_batch:
        batch = np.random.choice(len(generator), 1)
        X, y = generator[batch]
        h = model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
    else:
        h = model.evaluate(generator, verbose=0, batch_size=batch_size)
    one = "[^]" if one_batch else "[*]"
    history = Map({'loss': h[0], 'rmse': h[1], 'mae': h[2]})
    log('result', f"Home {home_id} || {dtype} {one} MSE: {h[0]:4f} | RMSE: {h[1]:4f}, MAE: {h[2]:4f}")
    return history


def get_params(model, named=False, numpy=False):
    if named:
        log("error", "Getting params using named params is not supported")
        exit()
    weights = [var.output_tensor for var in model.trainable_variables]
    return weights


def set_params(model, params, named=False, numpy=None):
    for i, param in enumerate(params):
        model.trainable_variables[i].output_tensor = param
    if named:
        log("error", "Setting params using named params is not supported")
        exit()


def GAR(peer, grads):
    if peer.params.gar == "average":
        return average(grads)
    elif peer.params.gar == "median":
        return median(grads)
    elif peer.params.gar == "aksel":
        return aksel(grads)
    elif peer.params.gar == "krum":
        return krum(grads)
    else:
        raise NotImplementedError()


def mape_metric(y_true, y_pred):
    """mean_absolute_percentage_error metric"""
    return K.mean(K.abs((y_true - y_pred) / y_true)) * 100


def me_metric(y_true, y_pred):
    """mean_error metric"""
    return K.mean(y_true - y_pred)


def mpe_metric(y_true, y_pred):
    """mean_percentage_error metric"""
    return K.mean((y_true - y_pred) / y_true) * 100


def timeseries_generator(X_train, X_test, Y_train, Y_test, length, batch_size=128, n_ahead=1):
    if n_ahead == 1:
        trainX, trainY = create_timeseries(Map({'X_test': X_train}), look_back=length, keep_dim=False)
        dataset = Map({'X_train': X_train, 'X_test': X_test})
        testX, testY = create_timeseries(dataset, look_back=length, keep_dim=True)
    else:
        trainX = trainY = testX = testY = None
        log("error", f"{n_ahead} step-ahead prediction is not allowed.")
        exit()

    return Map({'X': trainX, 'Y': trainY}), Map({'X': testX, 'Y': testY})


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


def load_model(*args):
    raise NotImplementedError()


def create_timeseries(dataset, look_back, keep_dim=True):
    dataX, dataY = [], []
    if keep_dim:
        test = np.concatenate((dataset.X_train[-look_back:], dataset.X_test), axis=0)
    else:
        test = dataset.X_test
    for i in range(len(test) - look_back):
        dataX.append(test[i:(i + look_back), :])
        dataY.append(test[i + look_back, -1])

    return np.array(dataX), np.array(dataY).reshape(-1, 1)


def model_predict(model, generator):
    X = generator['X']
    test_size = len(X)
    log('info', f"Prediction for {test_size} entries...")
    preds = model.predict(X, is_training=False).flatten()
    return preds


def n_steps_model_predict(model, dataset, steps, use_pred=True):
    pass
