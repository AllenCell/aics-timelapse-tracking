from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import tifffile

from timelapsetracking.dir_to_nodes import dir_to_nodes, img_to_nodes


def _gen_dummy_image(
        centroids: Sequence[Sequence[int]],
        shape=Sequence[int],
        first_label: int = 1,
) -> np.ndarray:
    """Generates a dummy image with objects specified by their centroids."""
    label = first_label
    img = np.zeros(shape, dtype=np.uint16)
    for centroid in centroids:
        slices = []
        for dim in centroid:
            start = dim - label
            width = 2*label + 1
            slices.append(slice(start, start + width))
        slices = tuple(slices)
        img[slices] = label
        label += 1
    return img


def _gen_dummy_image_dir(
        path_save_dir: Path,
        centroid_lists: Sequence[Sequence[Sequence[int]]],
        shape: Sequence[int],
) -> None:
    """Generates directory of dummy images."""
    path_save_dir.mkdir(parents=True, exist_ok=True)
    first_label = 1
    for idx, centroids in enumerate(centroid_lists):
        img = _gen_dummy_image(
            centroids=centroids, shape=shape, first_label=first_label
        )
        first_label += len(centroids)
        path_tif = path_save_dir / f'{idx:03d}_frame.tif'
        tifffile.imsave(path_tif, img)


def test_img_to_nodes():
    """Tests img_to_nodes function with dummy 2d and 3d images."""
    # Test bad inputs
    with pytest.raises(ValueError):
        img_to_nodes(np.zeros((10,), dtype=np.uint8))
    with pytest.raises(TypeError):
        img_to_nodes(np.zeros((10, 12), dtype=float))

    # Test blank image
    img = _gen_dummy_image(centroids=[], shape=(32, 32))
    result = img_to_nodes(img)
    assert not result

    # Test 2d image
    centroids = [(3, 4), (16, 19), (80, 80)]
    img = _gen_dummy_image(centroids=centroids, shape=(128, 256))
    df = pd.DataFrame(img_to_nodes(img))
    assert len(df) == 3
    centroid_y_exp = [3., 16., 80.]
    centroid_x_exp = [4., 19., 80.]
    label_img_exp = [1, 2, 3]
    volume_exp = [(label*2 + 1)**2 for label in label_img_exp]
    df_exp = pd.DataFrame({
        'centroid_y': centroid_y_exp,
        'centroid_x': centroid_x_exp,
        'label_img': label_img_exp,
        'volume': volume_exp,
    })
    assert df_exp.equals(df.filter(df_exp.columns))

    # Test 3d image
    centroids = [(2, 4, 8), (64, 128, 256)]
    img = _gen_dummy_image(centroids=centroids, shape=(128, 256, 512))
    df = pd.DataFrame(img_to_nodes(img))
    assert len(df) == 2
    centroid_z_exp = [2., 64.]
    centroid_y_exp = [4., 128.]
    centroid_x_exp = [8., 256.]
    label_img_exp = [1, 2]
    volume_exp = [(label*2 + 1)**3 for label in label_img_exp]
    df_exp = pd.DataFrame({
        'centroid_z': centroid_z_exp,
        'centroid_y': centroid_y_exp,
        'centroid_x': centroid_x_exp,
        'label_img': label_img_exp,
        'volume': volume_exp,
    })
    assert df_exp.equals(df.filter(df_exp.columns))


def test_dir_to_nodes(tmp_path):
    """Tests node generation from a directory of dummy images."""
    centroid_lists = [
        [],
        [(3, 4, 5), (80, 80, 80)],
        [(30, 60, 120), (64, 128, 256)],
    ]
    _gen_dummy_image_dir(
        path_save_dir=tmp_path,
        centroid_lists=centroid_lists,
        shape=(128, 256, 512),
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
