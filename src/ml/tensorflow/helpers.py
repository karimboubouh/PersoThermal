import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import layers as L
from keras.models import Sequential
from keras.saving.save import load_model
from tqdm import tqdm
from tqdm.keras import TqdmCallback

from .aggregators import average, median, aksel, krum
from ...helpers import Map
from ...utils import log


def initialize_models(model_name, input_shape, n_outputs=1, cpu=False, nbr_models=1, same=False):
    models = []
    if same:
        # Initialize all models with same weights
        model, custom_metrics = build_model(model_name, input_shape, n_outputs)
        if nbr_models == 1:
            models.append(model)
        else:
            model_file = f"./{model_name}.h5"
            model.save(model_file)
            for i in range(nbr_models):
                models.append(load_model(model_file, custom_objects=custom_metrics))
    else:
        # Independent initialization
        for i in range(nbr_models):
            model, _ = build_model(model_name, input_shape, cpu)
            models.append(model)

    return models


def build_model(model_name, input_shape, n_outputs=1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    rmse = tf.keras.metrics.RootMeanSquaredError(name='rmse')
    custom_metrics = {}
    metrics = [rmse, 'mae']
    model = Sequential()
    if model_name == 'RNN':
        model.add(L.SimpleRNN(100, activation='tanh', input_shape=input_shape))
    elif model_name == 'LSTM':
        model.add(L.LSTM(100, activation='tanh', input_shape=input_shape))
        # if cpu:
        #     model.add(L.LSTM(100, activation='relu', input_shape=input_shape))
        # else:
        #     model.add(L.CuDNNLSTM(100, input_shape=input_shape))
    elif model_name == 'DNN':
        model.add(L.Dense(100, activation='tanh', input_shape=(input_shape[1],)))
    elif model_name == 'BNN':
        raise NotImplementedError()
    else:
        exit('Error: Unrecognized model')
    # model.add(L.Dropout(0.2))
    model.add(L.Dense(n_outputs))
    model.compile(optimizer='adam', loss='mse', metrics=metrics)
    # print(model.summary())
    return model, custom_metrics


def model_fit(peer, tqdm_bar=False):
    train = peer.dataset.generator.train
    val = peer.dataset.generator.test
    if tqdm_bar:
        # , validation_data = val
        peer.model.fit(train, epochs=peer.params.epochs, batch_size=peer.params.batch_size, verbose=0,
                       callbacks=[TqdmCallback(verbose=2)])
    else:
        # , validation_data=val
        peer.model.fit(train, epochs=peer.params.epochs, validation_data=val, batch_size=peer.params.batch_size,
                       verbose=1)
    history = peer.model.history.history
    h = list(history.values())
    log('result',
        f"Node {peer.id} Train MSE: {h[0][-1]:4f}, RMSE: {h[1][-1]:4f} | MAE {h[2][-1]:4f}")

    return history


def meta_train(i, model_file, train, epochs=1):
    log('log', f"Node {i} performs personalized learning using local data for {epochs} epochs...")
    model = load_model(model_file)
    model.fit(train, epochs=epochs, verbose=0)
    history = model.history.history
    h = list(history.values())
    log('success', f"Node {i} META ML Train MSE: {h[0][-1]:4f} | RMSE: {h[1][-1]:4f} | MAE {h[2][-1]:4f}")
    # Node 4 META ML Train MSE
    # Node 4 Inference [*] MSE
    # Node 4 Fed Model Tst MSE

    return model, history


def meta_train2(i, model_file, train, epochs=1):
    log('event', f"Node {i} performs personalized learning using local data for {epochs} epochs...")
    model = load_model(model_file)
    history = Map({'loss': [], 'rmse': [], 'mae': []})
    h = [0, 0, 0]
    for epoch in range(epochs):
        X, y = train[-epochs + epoch - 1]
        h = model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
        history['loss'].append(h[0])
        history['rmse'].append(h[1])
        history['mae'].append(h[2])
    log('success', f"Node {i} META Train MSE: {h[0]:4f} | RMSE: {h[1]:4f} | MAE {h[2]:4f}")

    return model, history


def train_for_x_epochs(peer, epochs=1, verbose=0, evaluate=False, use_tqdm=False):
    h1 = Map({'loss': [], 'rmse': [], 'mae': []})
    h2 = None
    train = peer.dataset.generator.train
    peer.model.fit(train, epochs=epochs, batch_size=peer.params.batch_size, verbose=verbose)
    h = list(peer.model.history.history.values())
    h1.loss.append(h[0])
    h1.rmse.append(h[1])
    h1.mae.append(h[2])
    if evaluate:
        test = peer.dataset.generator.test
        h = peer.model.evaluate(test, verbose=verbose, batch_size=peer.params.batch_size)
        h2 = Map({'loss': h[0], 'rmse': h[1], 'mae': h[2]})

    return Map({'train': h1, 'test': h2})


def train_for_x_batches(peer, batches=1, evaluate=False, use_tqdm=True):
    h1 = Map({'loss': [], 'rmse': [], 'mae': []})
    h2 = None
    T = tqdm(range(batches), position=0) if use_tqdm else range(batches)
    for _ in T:
        train = peer.dataset.generator.train
        batch = np.random.choice(len(train), 1)
        X, y = train[batch]
        h = peer.model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
        # if h[1] > 1:
        #     log('error', f"{peer} | h={h} | y={y}, batch={batch}")
        h1.loss.append(h[0])
        h1.rmse.append(h[1])
        h1.mae.append(h[2])

    if evaluate:
        test = peer.dataset.generator.test
        batch = np.random.choice(len(test), 1)
        X, y = test[batch]
        h2 = peer.model.test_on_batch(X, y, reset_metrics=False, return_dict=True)
        if h2[1] > 1:
            log('error', f"{peer} | h={h} | y={y}, batch={batch}")

    return Map({'train': h1, 'test': h2})


def model_inference(peer, one_batch=False):
    test = peer.dataset.generator.test
    if one_batch:
        batch = np.random.choice(len(test), 1)
        X, y = test[batch]
        h = peer.model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
    else:
        h = peer.model.evaluate(test, verbose=0)
    history = Map({'loss': h[0], 'rmse': h[1], 'mae': h[2]})
    one = "[^]" if one_batch else "[*]"
    log('result', f"Node {peer.id} Inference {one} MSE: {h[0]:4f} | RMSE: {h[1]:4f}, MAE: {h[2]:4f}")
    return history


def evaluate_model(peer, one_batch=False, batch_size=64, verbose=False):
    test = peer.dataset.generator.test
    if one_batch:
        # TODO Correct | perform test on one batch of test not train_on_batch
        batch = np.random.choice(len(test), 1)
        X, y = test[batch]
        h = peer.model.train_on_batch(X, y, reset_metrics=False, return_dict=False)
    else:
        h = peer.model.evaluate(test, verbose=verbose)

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
        return {layer.name: layer.get_weights() for layer in model.layers}
    else:
        return model.get_weights()


def set_params(model, params, named=False, numpy=None):
    if named:
        log("error", "Setting params using named params is not supported")
        exit()
    else:
        model.set_weights(params)


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
    train_generator = None
    test_generator = None
    TG = tf.keras.preprocessing.sequence.TimeseriesGenerator
    if n_ahead == 1:
        train_generator = TG(X_train, Y_train, length=length, batch_size=batch_size)
        Xt = np.vstack((X_train[-length:], X_test))
        yt = np.vstack((Y_train[-length:], Y_test))
        test_generator = TG(Xt, yt, length=length, batch_size=batch_size)
    elif n_ahead > 1:
        TRY = np.array([Y_train[i: i + n_ahead].flatten() for i in range(len(Y_train) - n_ahead + 1)])
        train_generator = TG(X_train[:-n_ahead + 1], TRY, length=length, batch_size=batch_size)
        Xt = np.vstack((X_train[-length:], X_test[:-n_ahead + 1]))
        yt = np.vstack((Y_train[-length:], Y_test))
        TSY = np.array([yt[i: i + n_ahead].flatten() for i in range(len(yt) - n_ahead + 1)])
        test_generator = TG(Xt, TSY, length=length, batch_size=batch_size)
    else:
        log("error", f"{n_ahead} step-ahead prediction is not allowed.")
        exit()

    return train_generator, test_generator


def model_predict(model, generator):
    preds = []
    test_size = len(generator)
    # test_size = 12
    log('info', f"Prediction for {test_size} entries...")
    for i in range(test_size):
        X = generator[i][0]
        if i % 10 == 0:
            loader = "\\" if i % 20 == 0 else "/"
            print(f"> {loader} {i}/{test_size} ...", end="\r")
        pred = model.predict(X, verbose=0).flatten()
        preds = np.append(preds, pred)
    print()

    return np.array(preds)


def n_steps_model_predict(model, dataset, steps, use_pred=True):
    n_input = C.LOOK_BACK * C.RECORD_PER_HOUR
    X_test, _ = create_timeseries(dataset, look_back=n_input, keep_dim=True)
    preds = []
    test_size = len(X_test)
    log('info', f"Prediction for {steps} steps in the test set out of {test_size}...")
    shape = (1,) + X_test[0].shape
    for i in range(steps):
        X = np.reshape(X_test[i], shape)
        if i % 10 == 0:
            loader = "\\" if i % 20 == 0 else "/"
            print(f"> {loader} {i}/{steps} ...", end="\r")
        pred = model.predict(X, batch_size=1, verbose=0).flatten()
        if use_pred and i + 1 < test_size:
            X_test[i + 1][-1][-1] = pred
        preds = np.append(preds, pred)
    print()

    return np.array(preds)
