import copy
import time

from kivymd.toast import toast

from .Model import Model
from .aggregators import average, median, aksel, krum
from .models import *
from .utils import flatten_grads, unflatten_grad


def initialize_models(args, same=False):
    models = []
    architecture = None
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            architecture = CNNMnist
        elif args.dataset == 'fmnist':
            pass
        elif args.dataset == 'cifar':
            pass
    elif args.model == 'mlp':
        # Multi-layer perceptron
        if args.dataset == 'mnist':
            architecture = FFNMnist
        elif args.dataset == 'cifar':
            toast(f"Model <MLP> is not compatible with <CIFAR> dataset.")
            exit(0)
        else:
            architecture = None  # MLP
    elif args.model == 'linear':
        architecture = None  # LogisticRegression
    else:
        exit('Error: unrecognized model')

    # model params
    name = args.model.upper()
    if args.model in ['cnn', 'mlp']:
        input_dim = [28, 28, 1]
    else:
        input_dim = [28, 28]

    if same:
        # Initialize all models with same weights
        model = Model(name=name, input_dim=input_dim).initial(architecture())
        for i in range(args.num_users):
            models.append(copy.deepcopy(model))
    else:
        # Independent initialization
        for i in range(args.num_users):
            model = Model(name=name, input_dim=input_dim).initial(architecture())
            models.append(model)

    return models


def model_fit(peer):
    peer.model.train(peer.train.dataset, peer.train.targets)
    peer.model.val(peer.val.dataset, peer.val.targets)
    # peer.model.test(peer.test.dataset, peer.test.targets)
    # todo REMOVE
    peer.model.test(peer.val.dataset, peer.val.targets)
    history = peer.model.fit(
        lr=peer.params.lr,
        momentum=peer.params.momentum,
        max_epoch=peer.params.epochs,
        batch_size=peer.params.batch_size,
        evaluation=True,
        logger=peer.log
    )
    return history


def train_for_x_epoch(peer, batches=1, evaluate=False):
    return peer.model.improve(batches, evaluate)


def evaluate_model(model, dataholder, one_batch=False):
    loss, acc = model.evaluate(dataholder.dataset, dataholder.targets, one_batch=one_batch)
    return {'val_loss': loss, 'val_acc': acc}


def model_inference(peer, one_batch=False):
    t = time.time()
    loss, acc = peer.model.evaluate(peer.inference.dataset, peer.inference.targets, one_batch)
    o = "I" if one_batch else "*"
    t = time.time() - t
    peer.log('result', f"{peer} [{t:.2f}s]{o} Inference loss: {loss:.4f},  acc: {(acc * 100):.2f}%")


def get_params(model, named=False):
    if named:
        return model.named_parameters()
    else:
        return model.parameters


def set_params(model, params, named=False):
    if named:
        toast("error", "Setting params using named params is not supported")
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
