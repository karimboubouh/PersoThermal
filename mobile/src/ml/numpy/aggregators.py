import numpy as np


def average(gradients):
    """ Aggregate the gradients using the average aggregation rule."""
    # Assertion

    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    if len(gradients) > 1:
        return np.mean(gradients, 0)
    else:
        return gradients[0]


def median(gradients):
    """ Aggregate the gradients using the median aggregation rule."""
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    return np.median(gradients, 0)


def aksel(gradients):
    """ Aggregate the gradients using the AKSEL aggregation rule."""
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    med = np.median(gradients, axis=0)
    matrix = gradients - med
    normsq = [np.linalg.norm(grad) ** 2 for grad in matrix]
    med_norm = np.median(normsq)
    correct = [gradients[i] for i, norm in enumerate(normsq) if norm <= med_norm]

    return np.mean(correct, axis=0)


def krum(gradients, f=1):
    """ Aggregate the gradients using the Krum aggregation rule."""
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    nbworkers = len(gradients)
    gradients = np.array(gradients)
    # Distance computations
    scores = []
    sqr_dst = []
    for i in range(nbworkers - 1):
        sqr_dst = []
        gi = gradients[i].reshape(-1, 1)
        for j in range(nbworkers - 1):
            gj = gradients[j].reshape(-1, 1)
            dst = np.linalg.norm(gi - gj) ** 2
            sqr_dst.append(dst)
        indices = list(np.argsort(sqr_dst)[:nbworkers - f - 2])
        sqr_dst = np.array(sqr_dst)
        scores.append(np.sum(sqr_dst[indices]))
    correct = np.argmin(scores)

    return gradients[correct]


if __name__ == '__main__':
    # g = [[1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 2, 3]]
    g1 = [[1., 3., 4.], [9., 7., 6.], [5., 5., 5.]]
    g2 = [[9., 7, 6.], [1., 3., 4.], [5., 5., 5.]]
    g = np.stack([g1, g2])
    tg = np.array(g)
    print(average(tg))
    print(median(tg))
    print(aksel(tg))
    print(krum(tg))
