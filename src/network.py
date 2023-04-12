import time
from itertools import combinations
from time import sleep
from typing import List
import numpy as np
from scipy.spatial import distance

from src.helpers import timeit
from src.p2p import Node, Graph
from src.utils import cluster_peers, similarity_matrix, node_topology, log


def central_graph(models):
    nb_nodes = len(models)
    similarities = np.zeros((nb_nodes, nb_nodes))
    # similarities[nb_nodes - 1] = similarities.T[nb_nodes - 1] = 1
    # similarities[nb_nodes - 1][nb_nodes - 1] = 0
    similarities[-1] = similarities.T[-1] = 1
    similarities[-1][-1] = 0
    adjacency = similarities != 0
    clusters = {0: np.arange(nb_nodes)}

    return {
        'clusters': clusters,
        'similarities': similarities,
        'adjacency': adjacency
    }


def random_graph(models, rho=0.2, cluster_enabled=False, k=2, data=None):
    if rho and rho < 0:
        log('warning', f"Generating a negative similarity matrix.")
    # prob_edge = 1, rnd_state = None
    nb_nodes = len(models)
    if cluster_enabled:
        clusters = cluster_peers(nb_nodes, k)
    else:
        clusters = {0: np.arange(nb_nodes)}
    if data:
        adjacency, similarities = similarity_matrix(models, clusters, rho, data=data)
    else:
        adjacency, similarities = similarity_matrix(nb_nodes, clusters, rho, data=data)

    return {
        'clusters': clusters,
        'similarities': similarities,
        'adjacency': adjacency
    }


def full_graph(models):
    nb_nodes = len(models)
    clusters = {0: np.arange(nb_nodes)}
    similarities = np.ones((nb_nodes, nb_nodes))
    similarities[np.diag_indices_from(similarities)] = 0
    return {
        'clusters': clusters,
        'similarities': similarities,
        'adjacency': similarities != 0
    }


def metadata_graph(metadata: dict, rho=0.2, dist="euclidian", limit=0, mode="inverse"):
    log('warning', f"Network topology: rho={rho} | limit={limit} | dist={dist}")
    matrix = []
    if dist == "euclidian":
        for home_id in metadata:
            matrix.append([np.linalg.norm(metadata[home_id] - metadata[neighbor_id]) for neighbor_id in metadata])
    elif dist == "cosine":
        for home_id in metadata:
            matrix.append([distance.cosine(metadata[home_id], metadata[neighbor_id]) for neighbor_id in metadata])
    elif dist == "hamming":
        for home_id in metadata:
            matrix.append([distance.hamming(metadata[home_id], metadata[neighbor_id]) for neighbor_id in metadata])
    else:
        raise NotImplemented()
    matrix = np.array(matrix)
    maxd = np.max(matrix)
    matrix[np.diag_indices_from(matrix)] = maxd

    normalized = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    # change rho from similarity to distance
    # rho = 1 - rho
    for n in normalized:
        min_v = np.min(n)
        min_i = list(n).index(min_v)
        n[n > rho] = 1
        if np.min(n) == 1:
            n[min_i] = min_v
    similarities = 1.0 - normalized

    if 0 < limit < len(similarities[0]):
        rate = len(similarities[0]) - limit
        for s in similarities:
            small_indices = s.argsort()[:rate]
            s[small_indices] = 0
    clusters = {0: np.arange(len(similarities[0]))}
    m = [sum(1 for x in s if x > 0) for s in similarities]
    log('event', f"Network density: Peers have on average {np.mean(m)}(+-{np.std(m):.2f}) neighbors.")
    return {
        'clusters': clusters,
        'similarities': similarities,
        'adjacency': similarities != 0
    }


def disconnected_graph(models):
    nb_nodes = len(models)
    clusters = {i: i for i in range(nb_nodes)}
    similarities = np.zeros((nb_nodes, nb_nodes))
    return {
        'clusters': clusters,
        'similarities': similarities,
        'adjacency': similarities != 0
    }


def random2_graph(models, train_ds):
    # test_ds, user_groups, prob_edge=1, cluster_data=True, rnd_state=None
    # similarities based on data /or/ random similarities based on clfs
    # generate adj matrix based on similarities
    # partition nodes if cluster_data is True

    nb_nodes = len(models)
    x = train_ds.data
    # y = train_ds.targets

    print(x[0].shape)
    print(nb_nodes)
    exit(0)

    # clustering
    # groups = partition(x, y, nb_nodes, cluster_data, random_state=rnd_state)
    # print(groups)
    # exit()
    # nodes = list()
    # for i in range(nb_nodes):
    #     n = Node(i, *groups[i])
    #     nodes.append(n)
    #
    # for i, n in enumerate(nodes):
    #     n = [n] + [nodes[j] for j in range(nb_nodes) if i != j and random() < prob_edge]
    #     n.set_neighbors(n, [1 / len(n)] * len(n))

    # return nodes


def datasim_network(data, sigma=0.2):
    """Compute graph matrices according to users data"""
    nb_nodes = len(data)

    pairs = combinations(range(nb_nodes), 2)
    similarities = np.zeros((nb_nodes, nb_nodes))
    # calculate similarity matrix
    for p in pairs:
        vec1 = np.hstack(data[p[0]])
        vec2 = np.hstack(data[p[1]])
        similarities[p] = similarities.T[p] = 1 - distance.cosine(vec1, vec2)
    # apply similarity threshold
    similarities[similarities < sigma] = 0
    # get adjacency matrix
    adjacency = similarities > 0

    return adjacency, similarities


@timeit
def network_graph(topology, models, dataset, home_ids, args, edge=None):
    nbr_nodes = len(home_ids)
    clustered = True if len(topology['clusters']) > 1 else False
    peers = list()
    t = time.time()
    # train, val, test = train_val_test(train_ds, user_groups[0], args)
    for i in range(nbr_nodes):
        neighbors_ids, sim = node_topology(i, topology)
        if edge and edge.is_edge_device(i):
            device_bridge = edge.populate_device(i, models[i], dataset[home_ids[i]], neighbors_ids, clustered, sim)
            device_bridge.neighbors_ids = neighbors_ids
            peers.append(device_bridge)
        else:
            peer = Node(i, models[i], dataset[home_ids[i]], neighbors_ids, clustered, sim, args)
            peer.start()
            peers.append(peer)
    connect_to_neighbors(peers)
    graph = Graph(peers, topology, dataset, args)
    log('info', f"Network Graph constructed in {(time.time() - t):.4f} seconds")

    # Transformations
    graph.set_inference(args)

    return graph


def connect_to_neighbors(peers: List[Node]):
    t = time.time()
    connected = True
    for peer in peers:
        neighbors = [p for p in peers if p.id in peer.neighbors_ids]
        for neighbor in neighbors:
            if not peer.connect(neighbor):
                log('error', f"{peer} --> {neighbor} Not connected")
                connected = False
        sleep(0.01)
    if connected:
        log('success', f"Peers successfully connected with their neighbors in {(time.time() - t):.4f} seconds.")
    else:
        log('error', f"Some peers could not connect with their neighbors.")


def connect_to_neighbors_2(peers: List[Node]):
    connected = True
    nthreads = 0
    pees = 0
    for peer in peers:
        neighbors = [p for p in peers if p.id in peer.neighbors_ids]
        pees += len(peer.neighbors_ids)
        for neighbor in neighbors:
            if not peer.connect(neighbor):
                log('error', f"{peer} --> {neighbor} Not connected")
                connected = False
        nthreads += len(peer.neighbors)
        log('warning', f"Current {pees} - {nthreads}")
        sleep(0.01)
    if connected:
        log('success', f"Peers successfully connected with their neighbors.")
    else:
        log('error', f"Some peers could not connect with their neighbors.")

    for peer in peers:
        log('event', peer.neighbors)
    exit()
