import torch

# def sinkhorn(M, *, max_iters=1000, tol=1e-6)


def _sinkhorn_iteration(K, a, b, u):
    v = b / (K.T @ u)
    u = a / (K @ v)
    return u, v


def _check_has_converged(K, b, u, v, tol):
    right_marginal = torch.einsum('i,ij,j->j', u, K, v)
    distsq = (right_marginal @ right_marginal
              + b @ b
              - 2 * right_marginal @ b)
    return distsq <= tol * tol * b.shape[-1]


def _sinkhorn_until_convergence(K, a, b, tol):
    u = torch.full_like(a, 1. / a.shape[-1])
    v = torch.full_like(b, 1. / b.shape[-1])

    while not _check_has_converged(K, b, u, v, tol):
        u, v = _sinkhorn_iteration(K, a, b, u)

    return u, v


def _sinkhorn(M, a, b, *, reg=1., tol=1e-6, grad_iters=10):
    K = torch.exp(M / reg)
    with torch.no_grad():
        u, v = _sinkhorn_until_convergence(K, a, b, tol)
    for _ in range(grad_iters):
        u, v = _sinkhorn_iteration(K, a, b, u)
    return K, u, v


def _distance_matrix_sq(a, b):
    dists = -2. * a @ b.T
    assert dists.shape == (a.shape[0], b.shape[0])
    dists = dists + torch.einsum('ij,ij->i', a, a)[:, None]
    dists = dists + torch.einsum('ij,ij->i', b, b)[None, :]
    return dists


def w2_euclidean(
    a, b,
    a_weights=None, b_weights=None,
    *,
    reg=1.,
    tol=1e-6, grad_iters=10,
):
    if len(a.shape) != 2:
        raise ValueError('a must be 2-dimensional')
    if len(b.shape) != 2:
        raise ValueError('b must be 2-dimensional')

    if a_weights is None:
        a_weights = torch.full((a.shape[0],), 1. / a.shape[0])
    if b_weights is None:
        b_weights = torch.full((b.shape[0],), 1. / b.shape[0])

    if len(a_weights.shape) != 1:
        raise ValueError('a_weights must be 1-dimensional')
    if len(b_weights.shape) != 1:
        raise ValueError('b_weights must be 1-dimensional')

    if a.shape[0] != a_weights.shape[0]:
        raise ValueError('a_weights must have the same number of points as a')
    if b.shape[0] != b_weights.shape[0]:
        raise ValueError('b_weights must have the same number of points as b')

    if not (a_weights >= 0.).all():
        raise ValueError('a_weights must be >= 0.')
    if not (b_weights >= 0.).all():
        raise ValueError('b_weights must be >= 0.')

    with torch.no_grad():
        a_weights_sum = torch.sum(a_weights)
        b_weights_sum = torch.sum(b_weights)
    if not torch.isclose(a_weights_sum, torch.tensor(1.)):
        raise ValueError('a_weights must sum to 1')
    if not torch.isclose(b_weights_sum, torch.tensor(1.)):
        raise ValueError('b_weights must sum to 1')

    if torch.isnan(a).any():
        raise ValueError('a cannot be NaN')
    if torch.isnan(b).any():
        raise ValueError('b cannot be NaN')

    M = _distance_matrix_sq(a, b)
    if torch.isnan(M).any():
        raise RuntimeError(
            'NaN values encountered in pairwise distance matrix')
    K, u, v = _sinkhorn(
        -M, a_weights, b_weights,
        reg=reg, tol=tol, grad_iters=grad_iters)
    cost = torch.einsum('ij,ij,i,j->', K, M, u, v)
    cost = torch.sqrt(cost)

    return cost


def jakub_loss(ax, ay, bx, by, k, s):
    k = torch.tensor(k)
    s = torch.tensor(s)
    ax = ax * torch.rsqrt(s)
    bx = bx * torch.rsqrt(s)
    ay = ay * torch.rsqrt(k)
    by = by * torch.rsqrt(k)
    a = torch.stack((ax, ay), axis=-1)
    b = torch.stack((bx, by), axis=-1)
    return w2_euclidean(a, b)
