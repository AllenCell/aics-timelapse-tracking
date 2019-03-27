"""Visualization utility functions."""


from typing import Dict, List, Optional, Sequence, Tuple, Union
import os

from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from scipy.ndimage.morphology import binary_erosion
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from timelapsetracking.util import images_from_dir


Pos = Tuple[float, float]
Color = Union[str, Sequence[float]]


def get_color_mapping(
        identifiers: List[int], cmap_mpl: str = 'hsv'
) -> Dict[int, Color]:
    """Creates mapping from identifiers to colors.

    Parameters
    ----------
    identifiers
        Unique, hashable identifiers.
    cmap_mpl
        Matplotlib ColorMap from which to choose colors.

    Returns
    -------
    Dict[int, Color]
        Mapping from identifiers to colors.

    """
    cm = matplotlib.cm.get_cmap(cmap_mpl)
    identifiers = set(identifiers)
    id_to_color = {}
    floaties = np.linspace(0., 1., len(identifiers))
    for idx, item in enumerate(identifiers):
        id_to_color[item] = cm(floaties[idx])
    return id_to_color


def save_tif(path_save: str, img: np.ndarray):
    """Utility function to consistently save uint8 tifs."""
    if img.dtype == bool:
        img = img.astype(np.uint8)*255
    assert np.issubdtype(img.dtype, np.uint8)
    tifffile.imsave(path_save, img, compress=2)
    print('Saved:', path_save)


def _overlay(img_b: np.ndarray, img_o: np.ndarray) -> np.ndarray:
    """Overlays image on another with zero meaning transparent."""
    assert np.issubdtype(img_b.dtype, np.unsignedinteger)
    assert np.issubdtype(img_o.dtype, np.unsignedinteger)
    mask = img_o > 0
    if mask.ndim > 2:
        mask = np.any(mask, axis=2)
    img_new = img_b.copy()
    img_new[mask] = img_o[mask]
    return img_new


def _blend_lighten_only(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Blends two images with 'lighten only' blend mode."""
    assert np.issubdtype(img_a.dtype, np.unsignedinteger)
    assert np.issubdtype(img_b.dtype, np.unsignedinteger)
    assert img_a.shape == img_b.shape
    return np.maximum(img_a, img_b)


def visualize_objects(
        points: Optional[Sequence[Pos]] = None,
        circles: Optional[Sequence[Pos]] = None,
        segments: Optional[Sequence[Tuple[Pos, Pos]]] = None,
        radius: int = 16,
        shape: Tuple[int, int] = (512, 512),
        colors_points: Union[List[Color], Color] = 'lime',
        colors_segments: Union[List[Color], Color] = 'red',
) -> np.ndarray:
    """Draw circle and line segment objects."""
    ylim, xlim = shape
    if circles is None:
        circles = []
    if segments is None:
        segments = []
    if isinstance(colors_points, list):
        assert len(colors_points) == len(points)
        colors_points = ['lime' if c is None else c for c in colors_points]
    if isinstance(colors_segments, list):
        assert len(colors_segments) == len(segments)
        colors_segments = ['red' if c is None else c for c in colors_segments]
    patches = []
    fig, ax = plt.subplots(facecolor='black')
    if points:
        points_y, points_x = [coord for coord in zip(*points)]
        ax.scatter(points_x, points_y, c=colors_points, s=64.0)
    for center in circles:
        kwargs = {
            'color': 'salmon',
            'fill': False,
            'linewidth': 1.0,
            'radius': radius,
        }
        if len(center) == 3 and isinstance(center[2], dict):
            kwargs.update(center[2])
        patches.append(
            Circle(
                (center[1], center[0]),
                **kwargs,
            )
        )
    for idx_seg, segment in enumerate(segments):
        ys, xs = [i for i in zip(*segment)]
        ax.plot(xs, ys, color=colors_segments[idx_seg], linewidth=4.0)
    dpi = fig.get_dpi()
    fig.set_size_inches(xlim/dpi, ylim/dpi)
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set_ylim(ylim, 0)
    ax.set_xlim(0, xlim)
    ax.set_axis_off()
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    fig.canvas.draw()
    ar = np.array(fig.canvas.renderer._renderer)[:, :, :3]
    plt.close(fig)
    return ar


def _pick_z_index(paths_tifs: List[str], n_check: int = 4):
    print('Picking z index....')
    n_check = min(n_check, len(paths_tifs))
    interval = len(paths_tifs)//n_check
    zs = []
    for idx_t in range(0, len(paths_tifs), interval)[:n_check]:
        print(idx_t)
        img = tifffile.imread(paths_tifs[idx_t])
        assert img.ndim == 3
        zs.append((img > 0).sum(axis=(1, 2)).argmax())
    return int(np.mean(zs).round())


def timelapse_3d_to_2d(
        path_in_dir: str, path_out_dir: str, val_object: int = 64
) -> None:
    """Converts 3d time-lapse tifs into 2d time-lapse tifs.

    Parameters
    ----------
    path_in_dir
        Directory of input 3d tifs.
    path_out_dir
        Directory to save output 2d tifs.
    val_object
        Value use to color each object.

    Returns
    -------
    None

    """
    paths_tifs = images_from_dir(path_in_dir)
    assert paths_tifs
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)
    z_index = _pick_z_index(paths_tifs)
    print('z_index:', z_index)
    dirname = os.path.basename(os.path.normpath(path_in_dir))
    for idx_t, path_tif in enumerate(paths_tifs):
        img = tifffile.imread(path_tif)
        assert img.ndim == 3
        img = (img[z_index, ] > 0).astype(np.uint8)*val_object
        img_line = np.logical_xor(img, binary_erosion(img, iterations=4))
        img[img_line] = 255
        path_save = os.path.join(
            path_out_dir, f'{dirname}.{idx_t:03d}.z{z_index}.tif'
        )
        save_tif(path_save, img)
    print('Done!!!')


class TrackVisualizer:
    """Visualize tracks from a sequence of centroids."""

    def __init__(
            self,
            shape: Tuple[int, int],
            alpha: float = 0.8,
    ):
        self.shape = shape
        self.alpha = alpha
        self.img_tracks_prev = np.zeros(self.shape + (3, ))

    def render(
            self,
            centroids: Sequence[Pos],
            segments: Sequence[Tuple[Pos, Pos]],
            colors_centroids: Optional[list] = None,
            colors_segments: Optional[list] = None,
    ) -> np.ndarray:
        """Render image of centroids with tracks.

        """
        img_centroids = visualize_objects(
            points=centroids,
            shape=self.shape,
            colors_points=colors_centroids,
        )
        img_tracks = _blend_lighten_only(
            (self.img_tracks_prev*self.alpha).round().astype(np.uint8),
            visualize_objects(
                segments=segments,
                shape=self.shape,
                colors_segments=colors_segments,
            )
        )
        self.img_tracks_prev = img_tracks
        return _overlay(img_tracks, img_centroids)


def visualize_tracks_2d(
        df: pd.DataFrame,
        shape: Tuple[int, int],
        path_save_dir: str,
        color_map: Optional[dict] = None,
) -> None:
    """

    Parameters
    ----------
    df
        DataFrame with 'centroid_y' and 'centroid_x' columns.
    shape
        Shape of output images.
    path_save_dir
        Directory to save images.
    color_map
        Mapping from track_id to color.

    """
    assert all(col in df.columns for col in ['centroid_y', 'centroid_x'])
    if color_map is None:
        color_map = {}
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
        print('Created:', path_save_dir)
    centroids = df.apply(
        lambda x: np.array([x['centroid_y'], x['centroid_x']]), axis=1
    )
    if color_map is not None:
        colors = df['track_id'].apply(lambda x: color_map.get(x))
    df = df.assign(centroid=centroids, color=colors)
    track_visualizer = TrackVisualizer(shape=shape)
    for group, df_g in df.groupby('index_sequence'):
        print('index_sequence:', group)
        segments = []
        colors_segments = []
        for _, row in df_g.iterrows():
            if not np.isnan(row.in_list):
                segments.append((
                    df.loc[int(row.in_list), 'centroid'],
                    row.centroid,
                ))
                colors_segments.append(color_map.get(row['track_id']))
        path_tif = os.path.join(path_save_dir, f'{group:03d}_tracks.tif')
        img = track_visualizer.render(
            centroids=df_g['centroid'].tolist(),
            segments=segments,
            colors_centroids=df_g['color'].tolist(),
            colors_segments=colors_segments,
        )
        tifffile.imsave(path_tif, img, compress=2)
        print('Saved:', path_tif)
    print('Done!!!')
