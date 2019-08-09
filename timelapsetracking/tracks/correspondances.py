import logging

from typing import List, Sequence, Tuple, Union, Optional

from scipy.optimize import linprog
from scipy.spatial import KDTree
import numpy as np


logger = logging.getLogger(__name__)


def _calc_dist(
        pos_a: np.ndarray, pos_b: np.ndarray
) -> Union[float, np.ndarray]:
    """Calculates the Euclidean distance between points."""
    axis_sum = 0 if (pos_a.ndim == pos_b.ndim == 1) else 1
    return (((pos_a - pos_b)**2).sum(axis=axis_sum))**0.5


def _similar_pairs(vec: np.ndarray, threshold: float):
    """Returns pairs of indices of input vector where values are similar
    within threshold.

    """
    maxes = np.maximum(vec[:, np.newaxis], vec[np.newaxis, ])
    diffs = vec[:, np.newaxis] - vec[np.newaxis, ]
    ndiffs = np.absolute(diffs)/maxes  # error relative to max
    return np.tril(ndiffs < threshold, -1)


def _calc_constraints(edges: Sequence[List]):
    pass


def _calc_pos_edges(
        centroids_a: np.ndarray,
        centroids_b: np.ndarray,
        thresh_dist: float,
        allow_splits: bool,
        cost_add: float,
        cost_delete: float,
        volumes_a: Optional[np.ndarray] = None,
        volumes_b: Optional[np.ndarray] = None,
) -> Tuple[Sequence[List], Sequence[float], Sequence[List], Sequence[List]]:
    """Find potential linkages between sets of centroids.

    Potential linkages are given a cost based on the Euclidean distance between
    the objects. In the case of object appearance or disappearance, cost is
    determined by cost_add or cost_delete respectively.

    Parameters
    ----------
    centroids_a
        Centroids of first set of objects.
    centroids_b
        Centroids of second set of objects.
    thresh_dist
        Maximum distance between potentially linked objects.
    allow_splits
        Set to allow object splits.
    cost_add
        Cost of object creation.
    cost_delete
        Cost of object deletion.
    volumes_a
    volumes_b

    Returns
    -------
    Tuple[Sequence[List], Sequence[float], Sequence[List], Sequence[List]]
        pos_edges, costs, indices_left, indices_right

    """
    if len(centroids_a) == 0 and len(centroids_b) == 0:
        return [], [], [], []
    elif len(centroids_a) > 0 and len(centroids_b) == 0:
        neighbors = [[] for _ in range(len(centroids_a))]
    elif len(centroids_a) == 0 and len(centroids_b) > 0:
        neighbors = []
    else:  # len(centroids_a) > 0 and len(centroids_b) > 0
        tree_a = KDTree(centroids_a)
        tree_b = KDTree(centroids_b)
        neighbors = tree_a.query_ball_tree(tree_b, thresh_dist)
    assert volumes_a is None or volumes_a.shape[0] == centroids_a.shape[0]
    assert volumes_b is None or volumes_b.shape[0] == centroids_b.shape[0]
    pos_edges = []
    indices_left = [[] for _ in range(len(centroids_a))]
    indices_right = [[] for _ in range(len(centroids_b))]
    costs = []
    for idx_a, indices_b in enumerate(neighbors):
        for idx_b in indices_b + [None]:  # None indicates "delete" vertex
            idx_e = len(pos_edges)
            indices_left[idx_a].append(idx_e)
            if idx_b is not None:
                indices_right[idx_b].append(idx_e)
                cost = _calc_dist(tree_a.data[idx_a], tree_b.data[idx_b])
            else:
                cost = cost_delete
            pos_edges.append((idx_a, idx_b))
            costs.append(cost)
        if allow_splits and len(indices_b) > 1:
            # Determine possible splits by two criteria: the angles and the
            # magnitudes of the displacement vectors. Two objects in the second
            # object group might be paired children if their displacement
            # vector angles are ~-180 apart and if the magnitudes are similar.
            displacements = tree_b.data[indices_b] - tree_a.data[idx_a]
            mags = np.sqrt(np.sum(displacements**2, axis=1))
            displacements /= mags[:, np.newaxis]
            maxes = np.maximum(mags[:, np.newaxis], mags[np.newaxis, ])
            diffs = mags[:, np.newaxis] - mags[np.newaxis, ]
            ndiffs = np.absolute(diffs)/maxes  # error relative to max
            angles = displacements.dot(displacements.T)
            # TODO: make thresholds function parameters
            masks = []
            masks.append(np.tril(ndiffs < 0.7, -1))
            masks.append(np.tril(angles < -0.85, -1))
            if volumes_b is not None:
                masks.append(_similar_pairs(volumes_b[indices_b], 0.3))
            mask = np.logical_and.reduce(masks)
            for indices_d in zip(*np.where(mask)):
                ordered = (
                    indices_d if indices_d[0] < indices_d[1] else
                    indices_d[1], indices_d[0]
                )
                pair = (indices_b[ordered[0]], indices_b[ordered[1]])
                cost_split = _calc_dist(
                    tree_a.data[idx_a], tree_b.data[pair, ]
                ).sum()
                idx_e = len(pos_edges)
                indices_left[idx_a].append(idx_e)
                indices_right[pair[0]].append(idx_e)
                indices_right[pair[1]].append(idx_e)
                pos_edges.append((idx_a, pair))
                costs.append(cost_split)

    # Add in "add" edges
    for idx_b in range(len(centroids_b)):
        idx_e = len(pos_edges)
        indices_right[idx_b].append(idx_e)
        pos_edges.append((None, idx_b))
        costs.append(cost_add)
    return pos_edges, costs, indices_left, indices_right


def find_correspondances(
        centroids_a: np.ndarray,
        centroids_b: np.ndarray,
        method_first: str = 'interior-point',
        thresh_dist: float = 45.,
        allow_splits: bool = True,
        cost_add: Optional[float] = None,
        cost_delete: Optional[float] = None,
        volumes_a: Optional[np.ndarray] = None,
        volumes_b: Optional[np.ndarray] = None,
):
    """Find correspondances from a to b using centroids.

    Parameters
    ----------
    centroids_a
        Array of object centroids from first group.
    centroids_b
        Array of object centroids from second group.
    method_first
        First method to try for linear programming solver.
    thresh_dist
        Maximum distance between potentially linked objects.
    allow_splits
        Set to allow object splits.
    cost_add
        Cost of object creation.
    cost_delete
        Cost of object deletion.
    volumes_a
    volumes_b

    """
    if centroids_a.ndim != 2 or centroids_b.ndim != 2:
        raise ValueError('Input centroids list must have dim 2')
    if centroids_a.shape[-1] != centroids_b.shape[-1]:
        raise ValueError(
            'Centroids in each input list must have the same dimensionality'
        )
    if method_first == 'simplex':
        methods = ['simplex', 'interior-point']
    elif method_first == 'interior-point':
        methods = ['interior-point', 'simplex']
    else:
        raise ValueError('Method first must be "simplex" or "interior-point"')
    cost_add = cost_add or thresh_dist*1.1
    cost_delete = cost_delete or thresh_dist*1.1
    pos_edges, costs, indices_left, indices_right = _calc_pos_edges(
        centroids_a=centroids_a,
        centroids_b=centroids_b,
        thresh_dist=thresh_dist,
        allow_splits=allow_splits,
        cost_add=cost_add,
        cost_delete=cost_delete,
        volumes_a=volumes_a,
        volumes_b=volumes_b,
    )
    if len(pos_edges) == 0:
        return []
    A_eq = []
    for indices in indices_left + indices_right:
        constraint = np.zeros(len(pos_edges), dtype=np.int)
        constraint[indices] = 1
        A_eq.append(constraint)
    b_eq = [1]*len(A_eq)
    for method in methods:
        result = linprog(
            c=costs,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0., 1.),
            method=method,
        )
        if result.success:
            break
        logger.info(f'Method {method} failed!')
    else:
        logger.info('Could not find solution!!!')
        raise ValueError
    indices = np.where(np.round(result.x))[0]
    edges = [pos_edges[idx] for idx in indices]
    return edges
