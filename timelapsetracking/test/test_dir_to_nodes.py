from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import tifffile

from timelapsetracking.dir_to_nodes import dir_to_nodes, img_to_nodes


def _gen_dummy_images(
        path_save_dir: Path,
        centroid_lists: Sequence[Sequence[int]],
        shape: Sequence[int],
) -> None:
    """Generate dummy images with object centroids specified by centroid_lists.

    Parameters
    ----------
    centroid_lists
        Sequence where each item is the object centroids for each frame.
    path_save_dir
        Dummy image save directory.
    shape
        Shape of the canvas for each frame.

    Returns
    -------
    None

    """
    path_save_dir.mkdir(parents=True, exist_ok=True)
    obj_id = 1
    for idx, cl in enumerate(centroid_lists):
        img = np.zeros(shape, dtype=np.uint16)
        for centroid in cl:
            slices = []
            for dim in centroid:
                start = dim - obj_id
                width = 2*obj_id + 1
                slices.append(slice(start, start + width))
            slices = tuple(slices)
            img[slices] = obj_id
            obj_id += 1
        path_tif = path_save_dir / f'{idx:03d}_frame.tif'
        tifffile.imsave(path_tif, img)


def test_dir_to_nodes_2d(tmp_path):
    """Tests node generation from dummy 2d images."""
    centroid_lists = [
        [(3, 4), (16, 19)],
        [(5, 5), (20, 20), (80, 80)],
    ]
    _gen_dummy_images(
        path_save_dir=tmp_path, centroid_lists=centroid_lists, shape=(128, 256)
    )
    df = dir_to_nodes(tmp_path)
    centroid_y_exp = [3., 16., 5., 20., 80.]
    centroid_x_exp = [4., 19., 5., 20., 80.]
    index_sequence_exp = [0, 0, 1, 1, 1]
    label_img_exp = [1, 2, 3, 4, 5]
    volume_exp = [(label*2 + 1)**2 for label in label_img_exp]
    df_exp = pd.DataFrame({
        'centroid_y': centroid_y_exp,
        'centroid_x': centroid_x_exp,
        'index_sequence': index_sequence_exp,
        'label_img': label_img_exp,
        'volume': volume_exp,
    })
    assert df_exp.equals(df.filter(df_exp.columns))


def test_dir_to_nodes_3d(tmp_path):
    """Tests node generation from dummy 3d images."""
    centroid_lists = [
        [],
        [(3, 4, 5), (80, 80, 80)],
        [(30, 60, 120), (64, 128, 256)],
    ]
    shape = (128, 256, 512)
    _gen_dummy_images(
        path_save_dir=tmp_path, centroid_lists=centroid_lists, shape=shape
    )
    df = dir_to_nodes(tmp_path)
    centroid_z_exp = [3., 80., 30., 64.]
    centroid_y_exp = [4., 80., 60., 128.]
    centroid_x_exp = [5., 80., 120., 256.]
    index_sequence_exp = [1, 1, 2, 2]
    label_img_exp = [1, 2, 3, 4]
    volume_exp = [(label*2 + 1)**3 for label in label_img_exp]
    df_exp = pd.DataFrame({
        'centroid_z': centroid_z_exp,
        'centroid_y': centroid_y_exp,
        'centroid_x': centroid_x_exp,
        'index_sequence': index_sequence_exp,
        'label_img': label_img_exp,
        'volume': volume_exp,
    })
    assert df_exp.equals(df.filter(df_exp.columns))


def test_img_to_nodes():
    """Tests bad inputs into img_to_nodes function."""
    with pytest.raises(ValueError):
        img_to_nodes(np.zeros((10,), dtype=np.uint8))
    with pytest.raises(TypeError):
        img_to_nodes(np.zeros((10, 12), dtype=float))
