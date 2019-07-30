"""Tests for tracks.edges module."""

import ast

import pandas as pd
import numpy as np

from timelapsetracking.tracks.edges import add_edges


def test_add_edges():
    """Test handling of dataframes with missing index_sequence values."""
    rng = np.random.RandomState(666)
    positions = rng.standard_normal((8, 2))
    frames = [0, 1, 2, 4, 5, 6, 8, 9]
    df = (
        pd.DataFrame(positions, columns=['centroid_y', 'centroid_x'])
        .rename_axis('node_id', axis=0)
        .assign(index_sequence=frames)
    )
    df = add_edges(df).sort_values('index_sequence')
    out_exp = [[1], [2], [], [4], [5], [], [7], []]
    out_got = [ast.literal_eval(i) for i in df['out_list'].tolist()]
    assert out_exp == out_got
