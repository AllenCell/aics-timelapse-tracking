from typing import List, Sequence, Tuple, Union, Optional

from scipy.optimize import linprog
from scipy.spatial import KDTree
import numpy as np


def _calc_dist(
        pos_a: np.ndarray, pos_b: np.ndarray
) -> Union[float, np.ndarray]:
    """Calculates the Euclidean distance between points."""
    axis_sum = 0 if (pos_a.ndim == pos_b.ndim == 1) else 1
    return (((pos_a - pos_b)**2).sum(axis=axis_sum))**0.5


def _calc_pos_edges(
        centroids_a: np.ndarray,
        centroids_b: np.ndarray,
        thresh_dist: float = 64.,
        allow_splits: bool = False,
        thresh_split: Optional[float] = None,
        cost_add: Optional[float] = None,
        cost_delete: Optional[float] = None,
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
    thresh_split
        Maximum distance between child objects after a split.
    cost_add
        Cost of object creation.
    cost_delete
        Cost of object deletion.

    Returns
    -------
    Tuple[Sequence[List], Sequence[float], Sequence[List], Sequence[List]]
        pos_edges, costs, indices_left, indices_right

    """
    if centroids_a.ndim != 2 or centroids_b.ndim != 2:
        raise ValueError('Input centroids list must have dim 2')
    if centroids_a.shape[-1] != centroids_b.shape[-1]:
        raise ValueError(
            'Centroids in each input list must have the same dimensionality'
        )
    if thresh_split is None:
        thresh_split = thresh_dist/2
    if cost_add is None:
        cost_add = thresh_dist*1.1
    if cost_delete is None:
        cost_delete = thresh_dist*1.1

    if len(centroids_a) == 0 and len(centroids_b) == 0:
        return []
    if len(centroids_a) > 0 and len(centroids_b) > 0:
        tree_a = KDTree(centroids_a)
        tree_b = KDTree(centroids_b)
        neighbors = tree_a.query_ball_tree(tree_b, thresh_dist)
    elif len(centroids_a) > 0 and len(centroids_b) == 0:
        neighbors = [[] for _ in range(len(centroids_a))]
    elif len(centroids_a) == 0 and len(centroids_b) > 0:
        neighbors = []
    else:
        raise AttributeError
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
        # Add edges for potential splits
        if allow_splits and len(indices_b) > 1:
            raise NotImplementedError('Splits not yet supported')
            tree_neighbors = KDTree(tree_b.data[indices_b, ])
            pairs = tree_neighbors.query_pairs(thresh_split)
            for pair in pairs:
                idx_e = len(pos_edges)
                cost = _calc_dist(
                    tree_a.data[idx_a], tree_neighbors.data[pair, ]
                ).mean()
                # Convert back original indices in tree_b
                indices_pair = (indices_b[pair[0]], indices_b[pair[1]])
                pos_edges.append((idx_a, indices_pair, cost))
                indices_left[idx_a].append(idx_e)
                indices_right[indices_pair[0]].append(idx_e)
                indices_right[indices_pair[1]].append(idx_e)
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
):
    """Find correspondances from a to b using centroids."""
    pos_edges, costs, indices_left, indices_right = _calc_pos_edges(
        centroids_a, centroids_b
    )
    if len(pos_edges) == 0:
        return []
    A_eq = []
    for indices in indices_left + indices_right:
        constraint = np.zeros(len(pos_edges), dtype=np.int)
        constraint[indices] = 1
        A_eq.append(constraint)
    b_eq = [1]*len(A_eq)
    for method in ['simplex', 'interior-point']:
        result = linprog(
            c=costs,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0., 1.),
            method=method,
        )
        if not result.success:
            print(f'Method {method} failed!')
        if result.success:
            break
    if not result.success:
        print('Could not find solution!!!')
        raise ValueError
    indices = np.where(np.round(result.x))[0]
    edges = [pos_edges[idx] for idx in indices]
    return edges
