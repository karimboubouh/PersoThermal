import torch
import torch.nn as nn
from math import pi


def angular_filter(peer, metric="cosine"):
    reference = peer.get_gradients().view(1, -1)
    accepted = []
    rejected = []
    for grad in peer.params.gradients.values():
        angle, distance = angular_metric(reference, grad.view(1, -1), metric)
        output = {'grad': grad, 'alpha': angle, 'gamma': distance}
        if angle <= peer.params.alpha_max:
            accepted.append(output)
        else:
            rejected.append(output)

    return accepted, rejected


def angular_metric(u, v, metric="cosine"):
    """
    Angular metric: calculates the angle and distance between two
    vectors using different distance metrics: euclidean, cosine,
    Triangle's Area Similarity (TS), Sector's Area Similarity (SS),
    and TS-SS. More details in the paper at
    https://github.com/taki0112/Vector_Similarity
    :param u: 1D Tensor
    :param v: 1D Tensor
    :param metric: choices are: cosine, euclidean, TS, SS, TS-SS
    :return: angle, distance
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(u, v)
    rad = torch.acos(similarity)
    angle = torch.rad2deg(rad).item()
    if metric == "cosine":
        distance = 1 - similarity.item()
    elif metric == "euclidean":
        distance = torch.cdist(u, v).item()
    elif metric == "TS":
        # Triangle's Area Similarity (TS)
        rad_10 = torch.deg2rad(rad + torch.deg2rad(torch.tensor(10.0)))
        distance = (torch.norm(u) * torch.norm(v)) * torch.sin(rad_10) / 2
        distance = distance.item()
    elif metric == "SS":
        # Sector's Area Similarity (SS)
        ed_md = torch.cdist(u, v) + torch.abs(torch.norm(u) - torch.norm(v))
        rad_10 = rad + torch.deg2rad(torch.tensor(10.0))
        distance = pi * torch.pow(ed_md, 2) * rad_10 / 360
        distance = distance.item()
    elif metric == "TS-SS":
        _, triangle = angular_metric(u, v, metric="TS")
        _, sector = angular_metric(u, v, metric="SS")
        distance = triangle * sector
    else:
        raise Exception(f"Distance metric {metric} unsupported")
    if similarity:
        distance = 1 - distance

    return angle, distance


if __name__ == '__main__':
    x = torch.Tensor([[4, 6, 8, 9, 2]])
    y = torch.Tensor([[40, 60, 80, 90, 20]])
    z = torch.Tensor([[1, -2, 16, 16, -1]])

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    a, b = angular_metric(x, y, metric="cosine")
    print(torch.cdist(x, y).item())
    print(b * torch.cdist(x, y).item())
    print("--")
    print(angular_metric(x, y, metric="cosine"))
    print(x - y)
    print(x - y - z)
    print(torch.cdist(x - y, z).item())
    print(angular_metric(x - y, z, metric="cosine"))
    exit()

    a, d = angular_metric(x, y, metric="cosine")
    print(f"cosine >> Distance = {d}, Angle = {a}°")

    a, d = angular_metric(x, y, metric="euclidean")
    print(f"euclidean >> Distance = {d}, Angle = {a}°")

    a, d = angular_metric(x, y, metric="TS")
    print(f"TS >> Distance = {d}, Angle = {a}°")

    a, d = angular_metric(x, y, metric="SS")
    print(f"SS >> Distance = {d}, Angle = {a}°")

    a, d = angular_metric(x, y, metric="TS-SS")
    print(f"TS-SS >> Distance = {d}, Angle = {a}°")
