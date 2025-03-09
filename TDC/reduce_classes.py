import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings

import math

from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures

from scipy.special import expit

from read_utils import get_mapping, read_gt_label, estimate_cost_matrix, avg_gt_activity_datasets

try:
    from pynndescent import NNDescent  # 能够正常使用pynndescent时就是True，我是安装了的应该是True

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

# use Approx NN to find first neighbor if samples more than ANN_THRESHOLD
ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine', use_tw_finch=False):
    s = mat.shape[0]

    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        if use_tw_finch:
            loc = mat[:, -1]
            mat = mat[:, :-1]
            loc_dist = np.sqrt((loc[:, None] - loc[:, None].T)**2)

        else:
            loc_dist = 1.

        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, 'cosine')
        orig_dist = orig_dist * loc_dist

        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError(
                "You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        if use_tw_finch:
            print(
                f'Since the video is larger than {ANN_THRESHOLD} samples, we cannot compute all distances. Instead FINCH will be used')
        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32),
                      (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)

    return A, orig_dist


def clust_rank_notime_wighted(mat, initial_rank=None, distance='cosine', use_tw_finch=False):
    s = mat.shape[0]

    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        if use_tw_finch:
            loc = mat[:, -1]
            mat = mat[:, :-1]
            loc_dist = np.sqrt((loc[:, None] - loc[:, None].T)**2)
        else:
            loc_dist = 1.

        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, 'cosine')

        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError(
                "You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        if use_tw_finch:
            print(
                f'Since the video is larger than {ANN_THRESHOLD} samples, we cannot compute all distances. Instead FINCH will be used')
        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32),
                      (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)

    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(
        csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'),
                         (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)  # c就是链接组件数量
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    r = d[idx]
    v = np.argsort(d[idx])
    v = v[:2]  # 只选了最小的两个
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    s = adj.get_shape()
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance, use_tw_finch=False):
    c_, mat = get_merge([], c, data)
    while(len(np.unique(c_)) > req_clust):
        adj, orig_dist = clust_rank(
            mat, initial_rank=None, distance=distance, use_tw_finch=use_tw_finch)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_, mat


def req_numclust_notime_weighted(c, data, req_clust, distance, use_tw_finch=False):
    c_, mat = get_merge([], c, data)
    while(len(np.unique(c_)) > req_clust):
        adj, orig_dist = clust_rank_notime_wighted(
            mat, initial_rank=None, distance=distance, use_tw_finch=use_tw_finch)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_, mat
