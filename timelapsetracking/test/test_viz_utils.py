from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tifffile

from timelapsetracking import viz_utils


plt.switch_backend('Agg')


def test_vizualize_tracks_2d(tmp_path):
    """Tests vizualize_tracks_2d function."""
    df_graph = pd.DataFrame(
        {
            'centroid_y': [6, 10, 14, 30, 25, 20],
            'centroid_x': [8, 18, 28, 60, 50, 40],
            'index_sequence': [0, 1, 2, 0, 1, 2],
            'lineage_id': [666, 666, 666, 777, 777, 777],
            'track_id': [666, 666, 666, 777, 777, 777],
        }
    ).rename_axis('node_id', axis=0)
    shape = (32, 64)
    with pytest.raises(ValueError):
        viz_utils.visualize_tracks_2d(
            df=df_graph, shape=shape, path_save_dir=tmp_path
        )
    df_graph['in_list'] = [str(x) for x in [[], [0], [1], [], [3], [4]]]
    last_frame = 7
    cm = viz_utils.get_color_mapping(df_graph['lineage_id'])
    viz_utils.visualize_tracks_2d(
        df=df_graph,
        shape=shape,
        path_save_dir=tmp_path,
        color_map=cm,
        index_max=last_frame,
    )
    assert len(list(tmp_path.glob('*'))) == last_frame + 1


def test_timelapse_3d_to_2d(tmp_path):
    """Tests timelapse_3d_to_2d function."""
    # Create dummy images
    path_in = tmp_path / Path('input')
    path_in.mkdir(parents=True, exist_ok=True)
    shape_img = (16, 32, 64)
    n_img = 8
    for idx in range(n_img):
        path_save = path_in / f'dummy_{idx:03d}.tif'
        img = np.random.randint(128, size=shape_img, dtype=np.uint16)
        tifffile.imsave(path_save, img)
    # Check for correct number of images in output directory
    path_out = tmp_path / Path('output')
    viz_utils.timelapse_3d_to_2d(path_in, path_out)
    assert path_out.exists()
    assert len(list(path_out.glob('*'))) == n_img
