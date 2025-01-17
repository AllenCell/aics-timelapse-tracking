import logging
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import KDTree

from timelapsetracking.metrics import calc_dist

logger = logging.getLogger(__name__)


def _similar_pairs(vec: np.ndarray, threshold: float):
    """Returns pairs of indices of input vector where values are similar
    within threshold.

    """
    maxes = np.maximum(vec[:, np.newaxis], vec[np.newaxis,])
    diffs = vec[:, np.newaxis] - vec[np.newaxis,]
    ndiffs = np.absolute(diffs) / maxes  # error relative to max
    return np.tril(ndiffs < threshold, -1)


def _calc_edge_groups(
    edges: List[Tuple],
    volumes_a: np.ndarray,
    volumes_b: np.ndarray,
    has_pair_b: np.ndarray,
    size_threshold: float = 0.70,
    weight_scaler: float = 2.0,
    pair_scaler: float = 0.25,
) -> List[List[int]]:
    """Calculate groups of edge indices that correspond to the same object.

    Parameters
    ----------
    edges
        List of proposed linkages between groups.

    Returns
    -------
    List[List[int]]
        List of edge groups.

    """
    constraints_a = defaultdict(list)
    constraints_b = defaultdict(list)
    weights_a = defaultdict(list)
    weights_b = defaultdict(list)

    pair_scales = np.ones_like(has_pair_b)
    pair_scales[has_pair_b] = pair_scaler

    for idx_e, (idx_a, idx_b) in enumerate(edges):
        w = 1

        if idx_b is not None:
            scale = pair_scales[idx_b]
        else:
            scale = 1

        if idx_a is not None and idx_b is not None:
            if isinstance(idx_b, int):
                w = volumes_a[idx_a] / volumes_b[idx_b]
            elif isinstance(idx_b, tuple):
                w = volumes_a[idx_a] / (volumes_b[idx_b[0]] + volumes_b[idx_b[1]])
        if idx_a is not None:
            constraints_a[idx_a].append(idx_e)
            if w < size_threshold or w > 1 / size_threshold:
                weights_a[idx_a].append((1 / w * scale) ** weight_scaler)
            else:
                weights_a[idx_a].append(1 * scale)

            if weights_a[idx_a][-1] < 0.2:
                weights_a[idx_a][-1] = 0.2
            if weights_a[idx_a][-1] > 5:
                weights_a[idx_a][-1] = 5

        if isinstance(idx_b, int):
            constraints_b[idx_b].append(idx_e)
            if w < size_threshold or w > 1 / size_threshold:
                weights_b[idx_b].append((w * scale) ** weight_scaler)
            else:
                weights_b[idx_b].append(1)

            if weights_b[idx_b][-1] < 0.2:
                weights_b[idx_b][-1] = 0.2
            if weights_b[idx_b][-1] > 5:
                weights_b[idx_b][-1] = 5
        elif isinstance(idx_b, tuple):
            constraints_b[idx_b[0]].append(idx_e)
            constraints_b[idx_b[1]].append(idx_e)
            if w < size_threshold or w > 1 / size_threshold:
                weights_b[idx_b[0]].append(w * scale)
                weights_b[idx_b[1]].append(w * scale)
            else:
                weights_b[idx_b[0]].append(1)
                weights_b[idx_b[1]].append(1)

            if weights_b[idx_b[0]][-1] < 0.2:
                weights_b[idx_b[0]][-1] = 0.2
            if weights_b[idx_b[1]][-1] < 0.2:
                weights_b[idx_b[1]][-1] = 0.2
            if weights_b[idx_b[0]][-1] > 5:
                weights_b[idx_b[0]][-1] = 5
            if weights_b[idx_b[1]][-1] > 5:
                weights_b[idx_b[1]][-1] = 5

    return (
        [c for c in constraints_a.values()] + [c for c in constraints_b.values()],
        [w for w in weights_a.values()] + [w for w in weights_b.values()],
    )


def _calc_pos_edges(
    centroids_a: np.ndarray,
    centroids_b: np.ndarray,
    thresh_dist: float,
    allow_splits: bool,
    cost_add: float,
    cost_delete: float,
    volumes_a: Optional[np.ndarray] = None,
    volumes_b: Optional[np.ndarray] = None,
) -> Tuple[Sequence[List], Sequence[float]]:
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
        Volumes for objects in first group.
    volumes_b
        Volumes for objects in second group.

    Returns
    -------
    Tuple[Sequence[List], Sequence[float]]
        pos_edges, costs

    """
    if len(centroids_a) == 0 or len(centroids_b) == 0:
        neighbors = [[] for _ in range(len(centroids_a))]
    else:  # len(centroids_a) > 0 and len(centroids_b) > 0
        tree_a = KDTree(centroids_a)
        tree_b = KDTree(centroids_b)
        neighbors = tree_a.query_ball_tree(tree_b, thresh_dist)
    assert volumes_a is None or volumes_a.shape[0] == centroids_a.shape[0]
    assert volumes_b is None or volumes_b.shape[0] == centroids_b.shape[0]

    pos_edges = []
    costs = []
    for idx_a, indices_b in enumerate(neighbors):
        for idx_b in indices_b + [None]:  # None indicates "delete" vertex
            if idx_b is not None:
                cost = calc_dist(tree_a.data[idx_a], tree_b.data[idx_b])
            else:
                cost = cost_delete  # [idx_a]
            pos_edges.append((idx_a, idx_b))
            costs.append(cost)

        if allow_splits and len(indices_b) > 1:
            # Determine possible splits by two criteria: the angles and the
            # magnitudes of the displacement vectors. Two objects in the second
            # object group might be paired children if their displacement
            # vector angles are ~-180 apart and if the magnitudes are similar.
            displacements = (tree_b.data[indices_b] - tree_a.data[idx_a]).astype(float)
            mags = np.sqrt(np.sum(displacements**2, axis=1)).astype(float)
            displacements /= mags[:, np.newaxis]
            maxes = np.maximum(mags[:, np.newaxis], mags[np.newaxis,])
            diffs = mags[:, np.newaxis] - mags[np.newaxis,]
            ndiffs = (
                np.absolute(diffs) / maxes
            )  # error relative to max# len(centroids_a) > 0 and len(centroids_b) > 0
            angles = displacements.dot(displacements.T)
            masks = []
            masks.append(np.tril(ndiffs < 0.7, -1))
            masks.append(np.tril(angles < -0.85, -1))

            if volumes_b is not None:
                masks.append(_similar_pairs(volumes_b[indices_b], 0.3))
            mask = np.logical_and.reduce(masks)
            for indices_d in zip(*np.where(mask)):
                ordered = (indices_d if indices_d[0] < indices_d[1] else indices_d[1], indices_d[0])
                pair = (indices_b[ordered[0]], indices_b[ordered[1]])
                cost_split = calc_dist(tree_a.data[idx_a], tree_b.data[pair,]).sum()
                pos_edges.append((idx_a, pair))
                costs.append(cost_split)

    # # Add in "add" edges
    for idx_b in range(len(centroids_b)):
        pos_edges.append((None, idx_b))
        costs.append(cost_add)  # [idx_b])
    return pos_edges, costs


def find_correspondances(
    centroids_a: np.ndarray,
    centroids_b: np.ndarray,
    method_first: str = "interior-point",
    thresh_dist: float = 45.0,
    thresh_size: float = 0.9,
    allow_splits: bool = False,
    cost_add: Optional[float] = None,
    cost_delete: Optional[float] = None,
    volumes_a: Optional[np.ndarray] = None,
    volumes_b: Optional[np.ndarray] = None,
    has_pair_b: Optional[np.ndarray] = None,
    is_bridge: Optional[bool] = False,
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
        Volumes for objects in first group.
    volumes_b
        Volumes for objects in second group.

    """
    if centroids_a.ndim != 2 or centroids_b.ndim != 2:
        raise ValueError("Input centroids list must have dim 2")
    if centroids_a.shape[-1] != centroids_b.shape[-1]:
        raise ValueError("Centroids in each input list must have the same dimensionality")
    if method_first == "simplex":
        methods = ["simplex", "interior-point"]
    elif method_first == "interior-point":
        methods = ["interior-point", "simplex"]
    else:
        raise ValueError('Method first must be "simplex" or "interior-point"')

    size_threshold = thresh_size
    if is_bridge:
        thresh_dist = thresh_dist * 1.5
        size_threshold = size_threshold * 0.9

    cost_add = cost_add or thresh_dist * 1.1
    cost_delete = cost_delete or thresh_dist * 1.1

    pos_edges, costs = _calc_pos_edges(
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
    (edge_groups, edge_weights) = _calc_edge_groups(
        pos_edges, volumes_a, volumes_b, has_pair_b, size_threshold
    )
    assert len(edge_groups) == (len(centroids_a) + len(centroids_b))
    assert len(edge_weights) == (len(centroids_a) + len(centroids_b))
    A_eq = []

    for _, (indices, weights) in enumerate(zip(edge_groups, edge_weights)):
        constraint = np.zeros(len(pos_edges), dtype=float)
        for _, (index, weight) in enumerate(zip(indices, weights)):
            constraint[index] = weight
        A_eq.append(constraint)

    b_eq = [1] * len(A_eq)
    for method in methods:
        result = linprog(
            c=costs,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0.0, 1.0),
            method=method,
        )

        if result.success:
            break
        # logger.info(f'Method {method} failed!')
    else:
        for _, (indices, weights) in enumerate(zip(edge_groups, edge_weights)):
            print(str(indices) + " -> " + str(weights))
        logger.info("Could not find solution!!!")
        raise ValueError
    indices = np.where(np.round(result.x))[0]
    edges = [pos_edges[idx] for idx in indices]

    # import pdb
    # pdb.set_trace()

    from itertools import combinations

    for i in range(2):
        for edge_1, edge_2 in combinations(edges, 2):
            # import pdb
            # pdb.set_trace()
            # if edge_1[1] == edge_2[1]:
            #     print(str(edge_1[1:]) + ' ' + str(edge_2[1:]))
            # if bool(set(edge_1[1:]) & set(edge_2[1:])): # and edge_1[0] != edge_2[0]:
            if any([i in edge_1[1:] for i in edge_2[1:]]):

                # import pdb
                # pdb.set_trace()

                if edge_1[-1] is None:
                    continue

                print("merge found")
                print(str(edge_1) + " " + str(edge_2))

                # if edge_1[0] is None:
                #     try:
                #         edges.remove(edge_1)
                #         continue
                #     except:
                #         continue

                # elif edge_2[0] is None:
                #     try:
                #         edges.remove(edge_2)
                #         continue
                #     except:
                #         continue

                idx_1 = pos_edges.index(edge_1)
                idx_2 = pos_edges.index(edge_2)

                cost_1 = result.x[idx_1]
                cost_2 = result.x[idx_2]

                print(str(cost_1) + " vs " + str(cost_2))
                print(cost_1 / cost_2)

                ratio = cost_1 / cost_2

                if ratio < 0.9:
                    try:
                        edges.remove(edge_1)
                        edges.append((edge_1[0], None))
                        print("Removed edge 1")
                    except:
                        continue
                else:
                    try:
                        edges.remove(edge_2)
                        edges.append((edge_2[0], None))
                        print("Removed edge 2")
                    except:
                        continue

    # import pdb
    # pdb.set_trace()

    return edges
