import time

import numpy as np

from .layers import *


class Model(object):
    def __init__(self, name, input_dim, n_class=None):
        self.name = name
        self.lr = 0.1
        self.log = None
        self.momentum = 0.9
        self.batch_size = 64
        self.__n_class = n_class
        self.__input_dim = input_dim
        self.__z = {}
        self.__e = {}
        self.__pool_index = {}
        self.__layer_block_dct = {}
        self.__layer_name_lst = ['x']
        self.__layer_output_dim_lst = [input_dim]

        self.__train_x_set = None
        self.__train_y_set = None
        self.__val_x_set = None
        self.__val_y_set = None
        self.__test_x_set = None
        self.__test_y_set = None

        self.__train_loss_log = []
        self.__train_acc_log = []

    def initial(self, block):
        temp_dim = self.__input_dim
        for i, layer_block in enumerate(block):
            name, temp_dim = layer_block.initial(temp_dim)
            if name not in self.__layer_name_lst:
                self.__layer_name_lst.append(name)
                self.__layer_output_dim_lst.append(temp_dim)
                self.__layer_block_dct[name] = layer_block
            else:
                print('Repeated Layer Name: {}!'.format(name))
                exit(1)
        # self.print_structure()
        return self

    def train(self, train_x_set, train_y_set):
        self.__train_x_set = train_x_set
        self.__train_y_set = train_y_set

    def val(self, val_x_set, val_y_set):
        self.__val_x_set = val_x_set
        self.__val_y_set = val_y_set

    def test(self, test_x_set, test_y_set):
        self.__test_x_set = test_x_set
        self.__test_y_set = test_y_set

    def print_structure(self):
        for i in range(len(self.__layer_name_lst)):
            print("{}:Layer[{}] Output Dim={}".format(self.name,
                                                      self.__layer_name_lst[i], self.__layer_output_dim_lst[i]))

    def named_parameters(self):
        params = {}
        for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
            if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                for k, v in layer_block.get_weights().items():
                    params[f"{layer_name}.{k}"] = v
        for elem in params.items():
            yield elem

    @property
    def parameters(self):
        params = []
        for layer_block in self.__layer_block_dct.values():
            if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                params.extend(layer_block.get_weights().values())
        return params

    @parameters.setter
    def parameters(self, params):
        i = 0
        for layer_block in self.__layer_block_dct.values():
            if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                layer_block.set_weights({'w': params[i], 'b': params[i + 1]})
                i += 2

    def fit(self, lr, momentum=0.9, max_epoch=1000, batch_size=128, shuffle=True, evaluation=False, logger=None):
        """
        Training model by SGD optimizer.
        :param lr: learning rate
        :param momentum: momentum rate
        :param max_epoch: max epoch
        :param batch_size: batch size
        :param shuffle: whether shuffle training set
        :param evaluation: evaluate model using eval set
        :param logger: logger function
        :return: none
        """
        history = []
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        if self.__train_x_set is None:
            print("None data fit!")
            exit(1)
        _vg = {}  # change to an init function
        for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
            if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                weight_shape = layer_block.weight_shape()
                _vg[layer_name] = \
                    {weight_name: np.zeros(list(weight_shape[weight_name])) for weight_name in weight_shape}
        batch_nums = len(self.__train_x_set) // self.batch_size

        for e in range(max_epoch):
            t = time.time()
            if shuffle:
                self.__shuffle_set(self.__train_x_set, self.__train_y_set)
            for i in range(batch_nums):
                start_index = i * self.batch_size
                sub = range(start_index, start_index + self.batch_size)
                t_x = self.__train_x_set[sub]
                t_y = self.__train_y_set[sub]
                _g, _batch_train_loss, _batch_train_acc = self.__gradient(t_x, t_y)
                for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
                    if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                        for weight_name in _g[layer_name]:
                            _vg[layer_name][weight_name] = self.momentum * _vg[layer_name][weight_name] \
                                                           - self.lr * _g[layer_name][weight_name]
                            _g[layer_name][weight_name] = -_vg[layer_name][weight_name]
                self.__gradient_descent(_g)

            if evaluation:
                # todo add eval loss
                # print the training log of whole training set rather than batch:
                t = time.time() - t
                loss, acc = self.evaluate(self.__val_x_set, self.__val_y_set)
                # todo save train loss/acc also
                self.__train_loss_log.append(_batch_train_loss)
                self.__train_acc_log.append(_batch_train_acc)
                logger('info', "Epoch [{}] [{:.2f}s], val_loss: {:.4f}, val_acc: {:.4f}".format(e, t, loss, acc))
                history.append({'val_loss': loss, 'val_acc': acc})

        return history

    def improve(self, batches=1, evaluation=False):
        """
        Improve the trained model with more rounds model by SGD optimizer.
        :param batches:
        :param evaluation: evaluate model using eval set
        :return: none
        """
        _vg = {}  # change to an init function
        for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
            if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                weight_shape = layer_block.weight_shape()
                _vg[layer_name] = \
                    {weight_name: np.zeros(list(weight_shape[weight_name])) for weight_name in weight_shape}

        for i in range(batches):
            batch_ids = np.random.choice(range(len(self.__train_x_set)), self.batch_size, replace=False)
            b_x = self.__train_x_set[batch_ids]
            b_y = self.__train_y_set[batch_ids]
            _g, _batch_train_loss, _batch_train_acc = self.__gradient(b_x, b_y)
            for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
                if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                    for weight_name in _g[layer_name]:
                        _vg[layer_name][weight_name] = self.momentum * _vg[layer_name][weight_name] \
                                                       - self.lr * _g[layer_name][weight_name]
                        _g[layer_name][weight_name] = -_vg[layer_name][weight_name]
            self.__gradient_descent(_g)

        if evaluation:
            loss, acc = self.evaluate(self.__val_x_set, self.__val_y_set)
            return {'val_loss': loss, 'val_acc': acc}

        return None

    def predict(self, _x_set):
        self.__forward(_x_set)
        return np.argmax(self.__z[self.__layer_name_lst[-1]], axis=-1)

    def evaluate(self, _x_set, _target_set, one_batch=False):
        if one_batch:
            ids = np.random.choice(range(len(_target_set)), self.batch_size, replace=False)
            _x_set = _x_set[ids]
            _target_set = _target_set[ids]

        self.__forward(_x_set)
        self.__backward(_target_set)
        loss = self.__loss_of_current() / len(_x_set)
        acc = 0
        for i in range(len(_x_set)):
            if np.argmax(self.__z[self.__layer_name_lst[-1]][i]) == np.argmax(_target_set[i]):
                acc += 1
        acc /= len(_x_set)

        return loss, acc

    def __forward(self, _x_set):
        temp_z_set = _x_set.copy()
        self.__z['x'] = temp_z_set
        for layer_block in self.__layer_block_dct.values():
            if isinstance(layer_block, (Conv2D, Dense, Flatten, Activation, BasicRNN)):
                temp_z_set = layer_block.forward(temp_z_set)
                self.__z[layer_block.name] = temp_z_set
            elif isinstance(layer_block, MaxPooling2D):
                temp_z_set, self.__pool_index[layer_block.name] = layer_block.forward(temp_z_set)
                self.__z[layer_block.name] = temp_z_set

    def __backward(self, _target_set):
        _y_set = self.__z[self.__layer_name_lst[-1]]
        self.__e[self.__layer_name_lst[-1]] = np.sum(-_target_set * np.log(_y_set + 1e-8))
        self.__e[self.__layer_name_lst[-2]] = self.__cross_entropy_cost(_y_set, _target_set)
        for i in range(len(self.__layer_name_lst) - 2, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_name_down = self.__layer_name_lst[i - 1]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense, Flatten, BasicRNN)):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set)
            elif isinstance(layer_block, MaxPooling2D):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set, self.__pool_index[layer_name])
            elif isinstance(layer_block, Activation):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set, self.__z[layer_name_down])

    def __gradient(self, _x_set, _target_set):
        # _dw = {}
        # _db = {}
        _g = {}
        self.__forward(_x_set)
        self.__backward(_target_set)
        _batch_train_loss = self.__loss_of_current() / len(_x_set)
        _batch_train_acc = 0
        for i in range(len(_x_set)):
            if np.argmax(self.__z[self.__layer_name_lst[-1]][i]) == np.argmax(_target_set[i]):
                _batch_train_acc += 1
        _batch_train_acc /= len(_x_set)
        for i in range(len(self.__layer_name_lst) - 1, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_name_down = self.__layer_name_lst[i - 1]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                _z_down = self.__z[layer_name_down]
                _e = self.__e[layer_name]
                # _dw[layer_name], _db[layer_name] = layer_block.gradient(_z_down, _e)
                _g[layer_name] = layer_block.gradient(_z_down, _e)

        return _g, _batch_train_loss, _batch_train_acc

    def __gradient_descent(self, _g):
        for i in range(len(self.__layer_name_lst) - 1, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense, BasicRNN)):
                layer_block.gradient_descent(_g[layer_name])

    @staticmethod
    def __cross_entropy_cost(_y_set, _target_set):
        prd_prb = _y_set.copy()
        if len(prd_prb) != len(_target_set):
            print("Cross entropy error!")
            exit(1)
        return prd_prb - _target_set

    @staticmethod
    def __shuffle_set(sample_set, target_set):
        index = np.arange(len(sample_set))
        np.random.shuffle(index)
        return sample_set[index], target_set[index]

    def __loss_of_current(self):
        return self.__e[self.__layer_name_lst[-1]]

    def __repr__(self):
        to_string = ""
        for i in range(len(self.__layer_name_lst)):
            to_string += f"{self.name}:Layer[{self.__layer_name_lst[i]}] Output Dim={self.__layer_output_dim_lst[i]}\n"

        return to_string

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    def model_test2():
        _model = Model(name='TEST', input_dim=[3, 10, 10])
        _model.initial(
            [
                Conv2D(name='C1', kernel_size=[3, 3], filters=16, padding='valid'),
                Activation(name='A1', method='relu'),
                MaxPooling2D(name='P1', pooling_size=[2, 2]),
                Conv2D(name='C2', kernel_size=[3, 3], filters=32, padding='valid'),
                Activation(name='A2', method='relu'),
                Flatten(name='flatten'),
                Dense(name='fc1', units=100),
                Activation(name='A3', method='relu'),
                Dense(name='fc2', units=10),
                Activation(name='A4', method='softmax'),
            ]
        )
        x_set = np.random.randn(2, 3, 10, 10)
        y_set = np.zeros([2, 10])
        y_set[0, 2] = 1
        y_set[1, 7] = 1

        _model.fit(x_set, y_set)
        _model.train(lr=0.01, batch_size=1, max_epoch=100, interval=1)
        print(_model.predict(x_set))


    model_test2()
