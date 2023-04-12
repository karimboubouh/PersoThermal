import copy
import time

from .aggregators import average, median, aksel, krum
from src.ml.pytorch.models import *


def initialize_models(args, same=False):
    # INITIALIZE PEERS MODELS
    models = []
    modelClass = None
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            modelClass = CNNMnist
        elif args.dataset == 'fmnist':
            modelClass = CNNFashionMnist
        elif args.dataset == 'cifar':
            modelClass = CNNCifar
    elif args.model == 'mlp':
        # Multi-layer perceptron
        if args.dataset == 'mnist':
            modelClass = FFNMnist
        elif args.dataset == 'cifar':
            log('error', f"Model <MLP> is not compatible with <CIFAR> dataset.")
            exit(0)
        else:
            modelClass = MLP
    elif args.model == 'linear':
        modelClass = LogisticRegression
    else:
        exit('Error: unrecognized model')

    if same:
        # Initialize all models with same weights
        if args.model == 'cnn':
            model = modelClass(args=args)
        else:
            len_in = 28 * 28
            model = modelClass(dim_in=len_in, dim_out=args.num_classes)
        for i in range(args.num_users):
            models.append(copy.deepcopy(model))
        return models

    else:
        # Independent initialization
        for i in range(args.num_users):
            if args.model == 'cnn':
                model = modelClass(args=args)
            else:
                len_in = 28 * 28
                model = modelClass(dim_in=len_in, dim_out=args.num_classes)
            models.append(model)

    for model in models:
        model.to(args.device)

    return models


def model_fit(peer):
    history = []
    optimizer = peer.params.opt_func(peer.model.parameters(), peer.params.lr)  # , 0.99
    for epoch in range(peer.params.epochs):
        t = time.time()
        for batch in peer.train:
            # Train Phase
            loss = peer.model.train_step(batch, peer.device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation Phase
        result = peer.model.evaluate(peer.val, peer.device)
        peer.model.epoch_end(epoch, result, time.time() - t)
        history.append(result)

    return history


def train_for_x_epoch(peer, batches=1, evaluate=False):
    for i in range(batches):
        # train for x batches randomly chosen when Dataloader is set with shuffle=True
        batch = next(iter(peer.train))
        # execute one training step
        optimizer = peer.params.opt_func(peer.model.parameters(), peer.params.lr)
        loss = peer.model.train_step(batch, peer.device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # get gradients
        # TODO review store gradients in "peer.grads"
        # grads = []
        # for param in peer.model.parameters():
        #     grads.append(param.grad.view(-1))
        # peer.grads = torch.cat(copy.deepcopy(grads))
    if evaluate:
        return peer.model.evaluate(peer.val, peer.device)

    return None


def evaluate_model(model, dataholder, one_batch=False, device="cpu"):
    return model.evaluate(dataholder, one_batch=one_batch, device=device)


def model_inference(peer, one_batch=False):
    t = time.time()
    r = peer.model.evaluate(peer.inference, peer.device, one_batch)
    o = "I" if one_batch else "*"
    acc = round(r['val_acc'] * 100, 2)
    loss = round(r['val_loss'], 2)
    t = round(time.time() - t, 1)
    log('result', f"Node {peer.id} [{t}s]{o} Inference loss: {loss}, acc: {acc}%")


def get_params(model, named=False, numpy=False):
    if named:
        return model.get_named_params(numpy=numpy)
    else:
        return model.get_params(numpy=numpy)


def set_params(model, params, named=False, numpy=False):
    if named:
        log("error", "Setting params using named params is not supported")
        exit()
    else:
        model.set_params(params, numpy=numpy)


def GAR(peer, grads, weighted=True):
    # Weighted Gradients Aggregation rule
    grads = torch.stack(grads)
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
