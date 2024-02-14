import numpy as np
import numpy.testing as npt
import pandas as pd

from timelapsetracking import metrics


def test__track_overlap():
    """Tests _track_overlap() function."""
    coords = [2.0**i for i in range(8)]
    df_b = pd.DataFrame(
        np.stack([coords, coords], axis=1),
        columns=["centroid_y", "centroid_x"],
        index=pd.Index(range(len(coords)), name="index_sequence"),
    ).assign(track_id=666)
    df_a = df_b.iloc[1 : len(coords) - 1].copy().assign(track_id=42)
    df_a.loc[:, ["centroid_y", "centroid_x"]] *= [1.1, 0.9]
    overlap = metrics._track_overlap(df_a=df_a, df_b=df_b, thresh_dist=32.0)
    assert overlap == 6  # 6 / 8 nodes covered

    # Threshold set to 0, so no nodes should overlap.
    overlap = metrics._track_overlap(df_a=df_a, df_b=df_b, thresh_dist=0.0)
    assert overlap == 0  # 0 / 8 nodes covered

    # Alter position of one node in track so it no longer covers the
    # ground-truth node.
    df_a.loc[df_a.index[2], "centroid_y"] *= 10
    overlap = metrics._track_overlap(df_a=df_a, df_b=df_b, thresh_dist=32.0)
    assert overlap == 5  # 5 / 8 nodes covered

    # Sequences indices shifted, so overlap should be 0
    df_a.index += df_b.shape[0]
    overlap = metrics._track_overlap(df_a=df_a, df_b=df_b, thresh_dist=32.0)
    assert overlap == 0  # 0 / 8 nodes covered


def test_target_effectiveness():
    # Create 2 ground-truth targets
    coords = [2.0**i for i in range(8)]
    df_targets = pd.DataFrame(
        np.stack([coords, coords, coords], axis=1),
        columns=["centroids_z", "centroid_y", "centroid_x"],
        index=range(len(coords)),
    ).assign(
        track_id=[3, 3, 3, 3, 3, 4, 4, 4],
        index_sequence=range(len(coords)),
    )

    # Create 3 predicted tracks
    coords_b = np.array([-10.0 * i for i in range(20, 24 + 1)])
    df_one_track = pd.DataFrame(
        np.stack([coords_b, coords_b * 1.1, coords_b * 0.9], axis=1),
        columns=["centroids_z", "centroid_y", "centroid_x"],
    ).assign(
        track_id=666,
        index_sequence=range(2, 2 + len(coords_b)),
    )
    df_two_tracks = df_targets.iloc[1:].copy()
    df_two_tracks["track_id"] *= 2  # change up the track_ids
    df_tracks = pd.concat([df_two_tracks, df_one_track]).reset_index(drop=True)

    te = metrics.target_effectiveness(
        df_tracks=df_tracks,
        df_targets=df_targets,
        thresh_dist=32,
    )
    te_exp = (4 + 3) / (5 + 3)
    npt.assert_approx_equal(te, te_exp)
