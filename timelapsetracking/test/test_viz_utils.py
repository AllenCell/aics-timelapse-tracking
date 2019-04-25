from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from timelapsetracking import viz_utils


plt.switch_backend('Agg')


def test_vizualize_tracks_2d():
    tmp_path = Path('tmp_for_test_delete_me')
    df_graph = pd.DataFrame(
        {
            'centroid_y': [6, 10, 14, 30, 25, 20],
            'centroid_x': [8, 18, 28, 60, 50, 40],
            'index_sequence': [0, 1, 2, 0, 1, 2],
            'track_id': [666, 666, 666, 777, 777, 777],
        }
    ).rename_axis('node_id', axis=0)
    shape = (32, 64)
    with pytest.raises(ValueError):
        viz_utils.visualize_tracks_2d(
            df=df_graph, shape=shape, path_save_dir=tmp_path
        )
    df_graph['in_list'] = [None, 0, 1, None, 3, 4]
    last_frame = 7
    cm = viz_utils.get_color_mapping(df_graph['track_id'])
    viz_utils.visualize_tracks_2d(
        df=df_graph,
        shape=shape,
        path_save_dir=tmp_path,
        color_map=cm,
        index_max=last_frame,
    )
    assert len([f for f in tmp_path.iterdir()]) == last_frame + 1
