from timelapsetracking.util import report_run_time
from typing import List, Optional
import numpy as np
import os
import pandas as pd
import pdb
import tifffile


def img_to_nodes(img: np.ndarray, meta: Optional[dict] = None) -> List[dict]:
    """Converts image of object labels to graph nodes.

    centroid_x, centroid_y, centroid_z, volume, metadata

    Parameters
    ----------
    img
        Image of labeled objects.

    Returns
    -------
    List[dict]
        List of dictionies representing each object.

    """
    assert img.ndim == 3
    assert np.issubdtype(img.dtype, np.unsignedinteger)
    if meta is None:
        meta = {}
    labels = np.unique(img)
    if labels[0] == 0:
        labels = labels[1:]
    nodes = []
    # TODO: add "label" to node information
    for label in labels:
        node = {}
        mask = img == label
        coords = np.where(mask)
        node['volume'] = mask.sum()
        for idx_d, dim in enumerate('zyx'):
            node[f'centroid_{dim}'] = coords[idx_d].mean()
        node.update(meta)
        nodes.append(node)
    return nodes


@report_run_time
def dir_to_nodes(path_dir: str) -> None:
    """Converts directory of label images to graph nodes.

    Parameters
    ----------
    path_dir
        Directory of images.

    Returns
    -------
    None

    """
    paths_img = sorted(
        [
            p.path for p in os.scandir(path_dir)
            if any(ext in p.path.lower() for ext in ['.tif', '.tiff'])
        ]
    )
    nodes = []
    for idx_s, path_img in enumerate(paths_img):
        print(f'Processing ({idx_s + 1}/{len(paths_img)}):', path_img)
        meta = {
            'path_tif': os.path.abspath(path_img),
            'index_sequence': idx_s,
        }
        nodes.extend(img_to_nodes(tifffile.imread(path_img), meta=meta))
    path_csv = os.path.join('tmp_outputs', 'nodes.csv')
    pd.DataFrame(nodes).rename_axis('node_id', axis=0).to_csv(path_csv)
    print('Saved:', path_csv)
