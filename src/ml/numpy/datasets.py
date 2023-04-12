import copy
import os
import struct
import time
from random import shuffle
from typing import Tuple, Any

import numpy as np
from numpy.random import multinomial

from src.conf import TRAIN_VAL_TEST_RATIO, DATASET_DUPLICATE
from src.utils import log

MNIST_PATH = r'./data/mnist/MNIST/raw'
CIFAR_PATH = r'./data/cifar/CIFAR/raw'


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    log('info', f"Loading {args.dataset.upper()} dataset ...")
    t = time.time()
    if args.dataset == 'mnist':
        train_dataset = MNIST(MNIST_PATH, train=True, shuffle=True)
        test_dataset = MNIST(MNIST_PATH, train=False, shuffle=False)
        if args.iid:
            if args.unequal:
                user_groups = mnist_iid_unequal(train_dataset, args.num_users)
            else:
                user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'cifar':
        raise NotImplementedError()

    return train_dataset, test_dataset, user_groups


# -- Sampling : MNIST ---------------------------------------------------------

class MNIST(object):
    def __init__(self, data_dir, train=None, shuffle=True, dimension=3):
        self.train = train
        self.train_x_set = None
        self.train_y_set = None
        self.train_labels_set = None

        self.test_x_set = None
        self.test_y_set = None
        self.test_labels_set = None

        self._shuffle = shuffle
        self.__dim = dimension
        if train is None or train is True:
            self.__load_mnist_train(data_dir)
        if train is None or train is False:
            self.__load_mnist_test(data_dir)

        self.__one_hot()
        self.__dimension()
        if self._shuffle:
            self.shuffle()
        self.__normalization()

    def __load_mnist_train(self, path, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        if DATASET_DUPLICATE > 0:
            labels = labels.repeat(DATASET_DUPLICATE, axis=0)
            images = images.repeat(DATASET_DUPLICATE, axis=0)
        self.train_x_set = images
        self.train_labels_set = labels

    def __load_mnist_test(self, path, kind='t10k'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        self.test_x_set = images
        self.test_labels_set = labels

    def __one_hot(self):
        if self.train in [None, True]:
            trn = np.zeros([len(self.train_labels_set), 10])
            for i, x in enumerate(self.train_labels_set):
                trn[i, x] = 1
            self.train_y_set = trn
        if self.train in [None, False]:
            te = np.zeros([len(self.test_labels_set), 10])
            for i, x in enumerate(self.test_labels_set):
                te[i, x] = 1
            self.test_y_set = te

    def __normalization(self, renormalize=False):
        mean = 0.1306604762738434
        std = 0.30810780385646314
        if self.train in [None, True]:
            self.train_x_set = self.train_x_set / 255.
            if self.__dim == 3 and renormalize is False:
                self.train_x_set -= mean
                self.train_x_set /= std

        if self.train in [None, False]:
            self.test_x_set = self.test_x_set / 255.
            if self.__dim == 3 and renormalize is False:
                self.test_x_set -= mean
                self.test_x_set /= std

        if self.__dim == 3 and renormalize is True:
            mean = 0
            std = 0
            for x in self.train_x_set:
                mean += np.mean(x[:, :, 0])
            mean /= len(self.train_x_set)
            self.train_x_set -= mean
            for x in self.train_x_set:
                std += np.mean(np.square(x[:, :, 0]).flatten())
            std = np.sqrt(std / len(self.train_x_set))
            print('The mean and std of MNIST:', mean, std)  # 0.1306604762738434 0.30810780385646314
            self.train_x_set /= std
            self.test_x_set -= mean
            self.test_x_set /= std

    def __dimension(self):
        if self.__dim == 1:
            pass
        elif self.__dim == 3:
            if self.train in [None, True]:
                self.train_x_set = np.reshape(self.train_x_set, [len(self.train_x_set), 28, 28, 1])
            if self.train in [None, False]:
                self.test_x_set = np.reshape(self.test_x_set, [len(self.test_x_set), 28, 28, 1])
        else:
            print('Dimension Error!')
            exit(1)

    def shuffle(self):
        if self.train in [None, True]:
            index = np.arange(len(self.train_x_set))
            np.random.shuffle(index)
            self.train_x_set = self.train_x_set[index]
            self.train_y_set = self.train_y_set[index]
            self.train_labels_set = self.train_labels_set[index]

        if self.train in [None, False]:
            index = np.arange(len(self.test_x_set))
            np.random.shuffle(index)
            self.test_x_set = self.test_x_set[index]
            self.test_y_set = self.test_y_set[index]
            self.test_labels_set = self.test_labels_set[index]

    @property
    def dataset(self):
        if self.train is None:
            return [self.train_x_set, self.test_x_set]
        elif self.train is True:
            return self.train_x_set
        elif self.train is False:
            return self.test_x_set

    @property
    def targets(self):
        if self.train is None:
            return [self.train_y_set, self.test_y_set]
        elif self.train is True:
            return self.train_y_set
        elif self.train is False:
            return self.test_y_set

    def __len__(self):
        if self.train is None:
            return len(self.train_x_set) + len(self.test_x_set)
        elif self.train is True:
            return len(self.train_x_set)
        elif self.train is False:
            return len(self.test_x_set)

    def __getitem__(self, index):
        if self.train is True:
            return self.train_x_set[index], self.train_y_set[index]
        elif self.train is False:
            return self.test_x_set[index], self.test_y_set[index]
        else:
            log('error', "Iterating MNIST dataset with Train=None is not supported")
            exit()

    def random_batches(self, batch_size, train=True, nb_batches=1):
        batches = []
        for i in range(nb_batches):
            ids = np.random.choice(range(len(self.train_x_set)), batch_size, replace=False)
            if self.train in [None, True] and train is True:
                batches.append((self.train_x_set[ids], self.train_y_set[ids]))
            elif self.train in [None, False] and train is False:
                batches.append((self.test_x_set[ids], self.test_y_set[ids]))
            else:
                raise "random_batches cannot return batches for train and test"
        return batches

    def slice(self, index):
        # new_self = copy.deepcopy(self)
        new_self = copy.copy(self)
        if self.train in [None, True]:
            new_self.train_x_set = self.train_x_set[index]
            new_self.train_y_set = self.train_y_set[index]
        if self.train in [None, False]:
            new_self.test_x_set = self.test_x_set[index]
            new_self.test_y_set = self.test_y_set[index]

        return new_self

    def __str__(self):
        return f"MNIST/Train={self.train} with {len(self)} samples"


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_iid_unequal(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    s = np.random.dirichlet(np.ones(num_users), size=1).flatten()
    num_items = multinomial(len(dataset), s)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items[i],
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_shards, num_imgs = 200, 300
    num_shards, num_imgs = estimate_shards(len(dataset), num_users)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = dataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # at least one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def mnist_noniid_unequal_org(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # at least one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


# -- Sampling : CIFAR ---------------------------------------------------------

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.train_labels)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


# -- Utility functions : ------------------------------------------------------

def estimate_shards(data_size, num_users):
    shards = num_users * 2 if num_users > 10 else 20
    imgs = int(data_size / shards)

    return shards, imgs


def train_val_test(train_ds, mask, args, ratio=None):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    ratio = TRAIN_VAL_TEST_RATIO if ratio is None else ratio
    mask = list(mask)
    shuffle(mask)
    assert np.sum(ratio) == 1, "Ratio between train, dev and test must sum to 1."
    v1 = int(ratio[0] * len(mask))
    v2 = int((ratio[0] + ratio[1]) * len(mask))
    # split indexes for train, validation, and test (80, 10, 10)
    train_mask = mask[:v1]
    val_mask = mask[v1:v2]
    test_mask = mask[v2:]

    # create data holders
    train_holder = train_ds.slice(train_mask)
    val_holder = train_ds.slice(val_mask)
    test_holder = train_ds.slice(test_mask)

    return train_holder, val_holder, test_holder


def inference_ds(peer, args):
    # todo handle inference scope
    if args.test_scope in ['global', 'neighborhood', 'local']:
        return peer.inference
    else:
        exit(f'Error: unrecognized TEST_SCOPE value: {args.test_scope}')


# -- MAIN TEST ----------------------------------------------------------------

if __name__ == '__main__':
    pass
    # from addict import Dict
    #
    # arguments = Dict({
    #     'dataset': 'mnist',
    #     'num_users': 10,
    #     'iid': True,
    #     'unequal': True,
    # })
    # a, b, c = get_dataset(arguments)
    # print(len(c[0]))

    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans)
    # num = 10
    # nds = dataset_train.data.numpy()
    # d = mnist_noniid_unequal(dataset_train, num)
    # print(f"Dataset shape: {nds.shape}")
    # print(f"({len(d)} users with data: {[len(i[1]) for i in d.items()]}")
