from typing import List, Optional, Tuple

import numpy as np
import numpy.testing as npt
import pytest

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
    for idx_t in range(4):
        if idx_t > 0:
            rng.shuffle(centroids_0)
            rng.shuffle(centroids_1)
        matching = find_correspondances(centroids_0, centroids_1)
        assert len(matching) == max(len(centroids_0), len(centroids_1))
        matching_as_centroids = []
        for idx_0, idx_1 in matching:
            # Use -1 to indicate coordinate of none existant centroid
            centroid_0 = centroids_0[idx_0] if idx_0 is not None else [-1]*nd
            centroid_1 = centroids_1[idx_1] if idx_1 is not None else [-1]*nd
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
    added = [[22, 33], [44, 55]]
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


@pytest.mark.skip(reason='Split handling currently not implemented')
def test_mitosis():
    """Test mitosis case."""
    import random
    random.seed(666)
    centroids_0 = [
        (16, 16),
        (16, 256),
        (256, 16),
        (256, 256),
    ]
    centroids_1 = []
    for idx_c in [1, 2]:
        centroid_new = tuple(
            coord + random.gauss(0, 4) for coord in centroids_0[idx_c]
        )
        centroids_1.append(centroid_new)
    centroid_new = tuple(
        coord + random.gauss(0, 4) for coord in centroids_0[1]
    )
    centroids_1.append(centroid_new)
    delta = 10
    centroids_1.append((256, 256 - delta))
    centroids_1.append((256, 256 + delta - 1))
    centroids_1.append((128, 128))  # Appearing centroid
    matches = find_correspondances(centroids_0, centroids_1)
    matches_exp = (
        (0, None), (1, (0, 2)), (2, 1), (3, 4), (None, 3), (None, 5)
    )
    assert _matchings_equal(matches, matches_exp)


if __name__ == '__main__':
    test_simple()
    test_delete()
    test_add()
    test_blanks()
    # test_mitosis()
