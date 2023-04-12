import numpy as np


def sigmoid(_x_set):
    return 1 / (1 + np.exp(-_x_set))


def d_sigmoid(_x_set):
    return (1 - sigmoid(_x_set)) * sigmoid(_x_set)


def softmax(_x_set):
    x_row_max = _x_set.max(axis=-1)
    x_row_max = x_row_max.reshape(list(_x_set.shape)[:-1] + [1])
    # print(f"_x_set.shape={_x_set.shape}/{np.sum(_x_set)}, x_row_max.shape={x_row_max.shape}/{np.sum(x_row_max)}")
    _x_set = _x_set - x_row_max
    x_exp = np.exp(_x_set)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(_x_set.shape)[:-1] + [1])
    return x_exp / x_exp_row_sum


def relu(_x_set):
    return np.maximum(0, _x_set)


def d_relu(_x_set):
    _x_set[_x_set <= 0] = 0
    _x_set[_x_set > 0] = 1
    return _x_set


def tanh(_x_set):
    return np.tanh(_x_set)


def d_tanh(_x_set):
    return 1 - np.tanh(_x_set) ** 2


def activation_function(method, x):
    if method == 'relu':
        _a = relu(x)
    elif method == 'sigmoid':
        _a = sigmoid(x)
    elif method == 'softmax':
        _a = softmax(x)
    elif method is None:
        _a = x
    else:
        _a = []
        print("No such activation: {}!".format(method))
        exit(1)
    return _a


def derivative_function(method, x):
    if method == 'relu':
        _d = d_relu(x)
    elif method == 'sigmoid':
        _d = d_sigmoid(x)
    elif method is None:
        _d = 1
    else:
        _d = []
        print("No such activation: {}!".format(method))
        exit(1)
    return _d


def _flatten(values):
    if isinstance(values, np.ndarray):
        # yield values.flatten()
        yield values.ravel()
    else:
        for value in values:
            yield from _flatten(value)


def flatten(values):
    # flatten nested lists of np.ndarray to np.ndarray
    return np.concatenate(list(_flatten(values)))


def _unflatten(flat_values, prototype, offset):
    if isinstance(prototype, np.ndarray):
        shape = prototype.shape
        new_offset = offset + np.product(shape)
        value = flat_values[offset:new_offset].reshape(shape)
        return value, new_offset
    else:
        result = []
        for value in prototype:
            value, offset = _unflatten(flat_values, value, offset)
            result.append(value)
        return result, offset


def unflatten(flat_values, prototype):
    # unflatten np.ndarray to nested lists with structure of prototype
    result, offset = _unflatten(flat_values, prototype, 0)
    assert (offset == len(flat_values))
    return result


def flatten_grads(grads):
    # flattened = np.array([np.concatenate([x.ravel() for x in g]) for g in grads])
    flattened = np.stack([flatten(grad) for grad in grads])
    return flattened


def unflatten_grad(grads, prototype):
    return unflatten(grads, prototype)
