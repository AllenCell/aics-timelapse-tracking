from typing import List, Optional, Tuple

import numpy as np
import numpy.testing as npt

from timelapsetracking.tracks.correspondances import find_correspondances


Pairing = Tuple[Optional[int], Optional[int]]


def _matchings_equal(
        matching_a: List[Pairing], matching_b: List[Pairing]
) -> bool:
    if len(matching_a) != len(matching_b):
        return False
    return all(
        matching_a[idx] == matching_b[idx] for idx in range(len(matching_a))
    )


def _get_centroid_sets(
        rng: Optional[np.random.RandomState]
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two sets of centroids for testing."""
    if rng is None:
        rng = np.random.RandomState(666)
    centroids_0 = np.array([
        (16, 16),
        (16, 32),
        (32, 64),
        (42, 42),
    ])
    centroids_1 = []
    for cen in centroids_0:
        centroid_new = tuple(coord + rng.normal(scale=4) for coord in cen)
        centroids_1.append(centroid_new)
    centroids_1 = np.array(centroids_1)
    return centroids_0, centroids_1


def _test_matches(
        centroids_0: np.ndarray,
        centroids_1: np.ndarray,
        first_matching: List[Pairing],
        rng: np.random.RandomState,
) -> None:
    """Find matches between two sets of centroids. The matching is done
    multiple times after shuffling the centroids.

    Parameters
    ----------
    centroids_0
        First set of centroids
    centroids_1
        Second set of centroids
    first_matching
        Expected matching without shuffling.
    rng
        RandomState for shuffling operation.

    """
    nd = centroids_0.shape[1]
    prev = None
    min_count = len(centroids_0)
    max_count = max(len(centroids_0), len(centroids_1))
    for idx_t in range(4):
        if idx_t > 0:
            rng.shuffle(centroids_0)
            rng.shuffle(centroids_1)
        matching = find_correspondances(centroids_0, centroids_1)
        assert min_count <= len(matching) <= max_count
        matching_as_centroids = []
        for idx_0, indices_1 in matching:
            if isinstance(indices_1, int) or indices_1 is None:
                indices_1 = [indices_1]
            # Use -1 to indicate coordinate of none existant centroid
            centroid_0 = centroids_0[idx_0] if idx_0 is not None else [-1]*nd
            for idx_1 in indices_1:
                centroid_1 = (
                    centroids_1[idx_1] if idx_1 is not None else [-1]*nd
                )
                matching_as_centroids.append(np.concatenate(
                    [centroid_0, centroid_1], axis=0
                ))
        matching_as_centroids = np.sort(
            np.array(matching_as_centroids), axis=0
        )
        if prev is None:
            assert _matchings_equal(matching, first_matching), (
                f'exp: {first_matching}, got: {matching}'
            )
        if prev is not None:
            npt.assert_array_almost_equal(prev, matching_as_centroids)
        prev = matching_as_centroids


def test_simple():
    """Test simple correspondance case with 4 centroids in each frame."""
    rng = np.random.RandomState(666)
    centroids_0, centroids_1 = _get_centroid_sets(rng)
    _test_matches(
        centroids_0, centroids_1, [(0, 0), (1, 1), (2, 2), (3, 3)], rng
    )


def test_delete():
    """Test correspondance when some centroids are deleted."""
    rng = np.random.RandomState(666)
    centroids_0, centroids_1 = _get_centroid_sets(rng)
    centroids_1 = centroids_1[1:-1, :]  # delete some centroids
    _test_matches(
        centroids_0, centroids_1, [(0, None), (1, 0), (2, 1), (3, None)], rng
    )


def test_add():
    """Test correspondance when some centroids are added."""
    rng = np.random.RandomState(666)
    centroids_0, centroids_1 = _get_centroid_sets(rng)
    added = [[128, 64], [128, 64]]
    centroids_1 = np.concatenate([centroids_1, added], axis=0)
    _test_matches(
        centroids_0=centroids_0,
        centroids_1=centroids_1,
        first_matching=[(0, 0), (1, 1), (2, 2), (3, 3), (None, 4), (None, 5)],
        rng=rng,
    )


def test_blanks():
    """Test correspondance when some centroids sets are empty."""
    rng = np.random.RandomState(666)
    centroids_0, centroids_1 = _get_centroid_sets(rng)
    nothing = np.zeros((0, centroids_0.shape[1]))
    _test_matches(
        centroids_0=centroids_0,
        centroids_1=nothing,
        first_matching=[(0, None), (1, None), (2, None), (3, None)],
        rng=rng,
    )
    _test_matches(
        centroids_0=nothing,
        centroids_1=centroids_1,
        first_matching=[(None, 0), (None, 1), (None, 2), (None, 3)],
        rng=rng,
    )


def test_mitosis():
    """Test mitosis case."""
    rng = np.random.RandomState(666)
    centroids_0, centroids_1 = _get_centroid_sets(rng)
    # Create additional child for objects 1 and 2 of first group
    children = []
    for cen in centroids_0[[1, 2], :]:
        children.append([coord + rng.normal(scale=4) for coord in cen])
    centroids_1 = np.concatenate((centroids_1, children))
    centroids_1 = np.concatenate((centroids_1, [[88, 66], [0, 128]]))
    _test_matches(
        centroids_0,
        centroids_1,
        [(0, 0), (1, (1, 4)), (2, (2, 5)), (3, 3), (None, 6), (None, 7)],
        rng,
    )
