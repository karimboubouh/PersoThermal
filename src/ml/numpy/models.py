from .layers import *

DNN_ARCH = [
    Flatten(name='flatten'),
    Dense(name='fc1', units=48),
    Activation(name='A1', method='relu'),
    Dense(name='fc2', units=64),
    Activation(name='A2', method='relu'),
    Dense(name='fc3', units=10),
    Activation(name='A3', method='softmax'),
]

CNN_ARCH = [
    Conv2D(name='C1', kernel_size=[3, 3], filters=5, padding='valid'),
    Activation(name='A1', method='relu'),
    MaxPooling2D(name='P1', pooling_size=[2, 2]),
    Flatten(name='flatten'),
    Dense(name='fc1', units=100),
    Activation(name='A3', method='relu'),
    Dense(name='fc2', units=10),
    Activation(name='A4', method='softmax'),
]

RNN_ARCH = [
    BasicRNN(name='R1', units=10, return_last_step=False),
    Activation(name='A1', method='relu'),
    BasicRNN(name='R2', units=10, return_last_step=False),
    Activation(name='A2', method='relu'),
    Flatten(name='flatten'),
    Dense(name='fc1', units=10),
    Activation(name='A3', method='softmax'),
]

LSTM_ARCH = []