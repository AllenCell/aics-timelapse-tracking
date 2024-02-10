from typing import Union

import numpy as np
import pandas as pd


def calc_dist(pos_a: np.ndarray, pos_b: np.ndarray) -> Union[float, np.ndarray]:
    """Calculates the Euclidean distance between points."""
    axis_sum = 0 if (pos_a.ndim == pos_b.ndim == 1) else 1
    return (((pos_a - pos_b) ** 2).sum(axis=axis_sum)) ** 0.5


def _track_overlap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    thresh_dist: float,
) -> int:
    """Counts number of overlapping (intersecting) nodes of two tracks.

    Parameters
    ----------
    df_a
        DataFrame representing a track.
    df_b
        DataFrame representing a track.
    thresh_dist
        Max distance between two node centroids to be considered a match.

    Returns
    -------
    int
        Number of overlapping nodes.

    """
    if not (df_a.index.name == df_b.index.name == "index_sequence"):
        raise ValueError("Input DataFrame should be indexed by index_sequence")
    df_isect = df_a.join(df_b, how="inner", lsuffix="_a", rsuffix="_b")
    cols_a = ["centroid_z_a", "centroid_y_a", "centroid_x_a"]
    cols_b = ["centroid_z_b", "centroid_y_b", "centroid_x_b"]
    centroids_a = df_isect.filter(cols_a).to_numpy()
    centroids_b = df_isect.filter(cols_b).to_numpy()
    assert centroids_a.shape == centroids_b.shape
    distances = calc_dist(centroids_a, centroids_b)
    return (distances <= thresh_dist).sum()


def target_effectiveness(
    df_tracks: pd.DataFrame,
    df_targets: pd.DataFrame,
    thresh_dist: float,
):
    """Computes the target effectiveness.

    target effectiveness = (
        sum(number of nodes where the best track follows a target) /
        number of node from all tracks
    )
    This is analogous to micro-averaged recall.

    Parameters
    ----------
    df_tracks
        DataFrame representing all tracks.
    df_targets
        DataFrame representing all targets (ground-truth tracks).
    thresh_dist
        Max distance between two node centroids to be considered a match.

    Returns
    -------
    float
        Target effectiveness, [0, 1.0]

    """
    cumulative_overlap = 0
    cumulative_target_length = 0
    for _, df_target in df_targets.groupby("track_id"):
        best = 0
        for _, df_track in df_tracks.groupby("track_id"):
            n_overlap = _track_overlap(
                df_a=df_target.set_index("index_sequence"),
                df_b=df_track.set_index("index_sequence"),
                thresh_dist=thresh_dist,
            )
            best = max(best, n_overlap)
        cumulative_overlap += best
        cumulative_target_length += df_target.shape[0]
    return cumulative_overlap / cumulative_target_length
