"""Visualization utility functions."""


from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import logging

from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from scipy.ndimage.morphology import binary_erosion
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from timelapsetracking.util import images_from_dir


plt.switch_backend('Agg')
Pos = Tuple[float, float]
Color = Union[str, Sequence[float]]
LOGGER = logging.getLogger(__name__)


def get_color_mapping(
        identifiers: Sequence[int], cmap_mpl: str = 'hsv'
) -> Dict[int, Color]:
    """Creates mapping from unique identifiers to colors.

    Parameters
    ----------
    identifiers
        Sequence of unique identifiers.
    cmap_mpl
        Matplotlib ColorMap from which to choose colors.

    Returns
    -------
    Dict[int, Color]
        Mapping from identifiers to colors.

    """
    cm = matplotlib.cm.get_cmap(cmap_mpl)
    id_to_color = {}
    floaties = np.linspace(0., 1., len(identifiers))
    for idx, item in enumerate(identifiers):
        id_to_color[item] = cm(floaties[idx])
    return id_to_color


def save_tif(path_save: str, img: np.ndarray) -> None:
    """Utility function to consistently save uint8 tifs.

    Parameters
    ----------
    path_save
        Path to save image.
    img
        Numpy array with dtype uint8 or bool.

    """
    if img.dtype not in [bool, np.uint8]:
        raise TypeError('Images must be type uint8 or bool')
    if img.dtype == bool:
        img = img.astype(np.uint8)*255
    tifffile.imsave(path_save, img, compress=2)
    LOGGER.info('Saved: %s', path_save)


def _overlay(img_b: np.ndarray, img_o: np.ndarray) -> np.ndarray:
    """Overlays image on another with zero meaning transparent."""
    if (
            not np.issubdtype(img_b.dtype, np.unsignedinteger)
            or not np.issubdtype(img_o.dtype, np.unsignedinteger)
    ):
        raise TypeError('Images must be np.unsignedinteger compatible')
    mask = img_o > 0
    if mask.ndim > 2:
        mask = np.any(mask, axis=2)
    img_new = img_b.copy()
    img_new[mask] = img_o[mask]
    return img_new


def _blend_lighten_only(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Blends two images with 'lighten only' blend mode."""
    if (
            not np.issubdtype(img_a.dtype, np.unsignedinteger)
            or not np.issubdtype(img_b.dtype, np.unsignedinteger)
    ):
        raise TypeError('Images must be np.unsignedinteger compatible')
    if img_a.shape != img_b.shape:
        raise ValueError("Images must be the same shape")
    return np.maximum(img_a, img_b)


def visualize_objects(
        points: Optional[Sequence[Pos]] = None,
        circles: Optional[Sequence[Pos]] = None,
        segments: Optional[Sequence[Tuple[Pos, Pos]]] = None,
        labels: Optional[Sequence[Tuple[Pos, str]]] = None,
        radius: int = 16,
        shape: Tuple[int, int] = (512, 512),
        colors_points: Union[List[Color], Color] = 'lime',
        colors_segments: Union[List[Color], Color] = 'red',
) -> np.ndarray:
    """Draws objects on a blank canvas.

    Parameters
    ----------
    points
        Points to draw.
    circles
        Circles to draw.
    segments
        Segments to draw.
    labels
        Labels to draw.
    radius
        Radius of drawn circles.
    shape
        Shape of canvas.
    colors_points
        Color(s) of drawn points.
    colors_segments
        Color(s) of drawn segments.

    Returns
    -------
    np.ndarray
        Image with drawn objects.

    """
    ylim, xlim = shape
    circles = circles or []
    segments = segments or []
    labels = labels or []
    if isinstance(colors_points, list):
        if len(colors_points) != len(points):
            raise ValueError('Length of color_points and points must match')
        colors_points = ['lime' if c is None else c for c in colors_points]
    if isinstance(colors_segments, list):
        if len(colors_segments) != len(segments):
            raise ValueError('Length of color_segments and segments must match')
        colors_segments = ['red' if c is None else c for c in colors_segments]
    patches = []
    fig, ax = plt.subplots(facecolor='black')
    if points:
        points_y, points_x = [coord for coord in zip(*points)]
        ax.scatter(points_x, points_y, c=colors_points, s=50.0)
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
        ax.plot(xs, ys, color=colors_segments[idx_seg], linewidth=6.0)
    for label in labels:
        y, x = label[0]
        ax.text(x=x, y=y, s=str(label[1]), color='white', fontweight='bold', fontsize=24)
    dpi = fig.get_dpi()
    fig.set_size_inches(xlim/dpi, ylim/dpi)
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set_ylim(ylim, 0)
    ax.set_xlim(0, xlim)
    ax.set_axis_off()
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    # Export figure to numpy array. Source: https://stackoverflow.com/a/7821917
    fig.canvas.draw()
    shape_fig = fig.canvas.get_width_height()[::-1] + (3,)
    ar = (
        np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        .reshape(shape_fig)
    )
    plt.close(fig)
    return ar


def _pick_z_index(paths_tifs: List[str], n_check: int = 4) -> int:
    """Selects a z index for transforming a 3d image sequence to a 2d image
    sequence.

    Parameters
    ----------
    paths_tifs
        Path to directory of tif images.
    n_check
        Maximum number of images to look at to determine returned z index.

    Returns
    -------
    int
        Selected z index.

    """
    LOGGER.info('Picking z index....')
    n_check = min(n_check, len(paths_tifs))
    interval = len(paths_tifs)//n_check
    zs = []
    for idx_t in range(0, len(paths_tifs), interval)[:n_check]:
        img = tifffile.imread(paths_tifs[idx_t])
        if img.ndim != 3:
            raise ValueError('Images must be 3d')
        zs.append((img > 0).sum(axis=(1, 2)).argmax())
    return int(np.mean(zs).round())


def timelapse_3d_to_2d(
        path_in_dir: Path, path_out_dir: Path, val_object: int = 64
) -> None:
    """Converts 3d time-lapse tifs into 2d time-lapse tifs.

    Parameters
    ----------
    path_in_dir
        Directory of input 3d tifs.
    path_out_dir
        Directory to save output 2d tifs.
    val_object
        Value used to color each object.

    Returns
    -------
    None

    """
    paths_tifs = images_from_dir(path_in_dir)
    if not paths_tifs:
        raise ValueError('TIF directory is empty.')
    path_out_dir.mkdir(parents=True, exist_ok=True)
    z_index = _pick_z_index(paths_tifs)
    LOGGER.info('Converting to 2d image sequence using z index %d', z_index)
    for idx_t, path_tif in enumerate(paths_tifs):
        img = tifffile.imread(path_tif)
        if img.ndim != 3:
            raise ValueError('Images must be 3d')
        img = (img[z_index, ] > 0).astype(np.uint8)*val_object
        img_line = np.logical_xor(img, binary_erosion(img, iterations=4))
        img[img_line] = 255
        path_save = Path(
            path_out_dir, f'{path_in_dir.name}.{idx_t:03d}.z{z_index}.tif'
        )
        save_tif(path_save, img)


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
            labels: Optional[list] = None,
    ) -> np.ndarray:
        """Render image of centroids with tracks.

        """
        if labels is not None:
            labels = zip(centroids, labels)
        img_centroids = visualize_objects(
            points=centroids,
            shape=self.shape,
            colors_points=colors_centroids,
            labels=labels,
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
        path_save_dir: Path,
        color_map: Optional[Dict] = None,
        index_max: Optional[int] = None,
        index_min: Optional[int] = None,
) -> None:
    """Generates an image sequence of object centroids and their tracking
    tails.

    Parameters
    ----------
    df
        Time-lapse graph DataFrame.
    shape
        Shape of output images.
    path_save_dir
        Directory to save images.
    color_map
        Mapping from track_id to color.
    index_max
        Index of last frame to render.
    index_min
        Index of first frame to render.

    """
    for col in [
            'centroid_y', 'centroid_x', 'index_sequence', 'track_id',
            'lineage_id', 'in_list',
    ]:
        if col not in df.columns:
            raise ValueError(f'{col} column expected in df')
    color_map = color_map or {}
    index_max = index_max or df['index_sequence'].max()
    index_min = index_min or df['index_sequence'].min()
    indices_todo = set(range(index_min, index_max + 1))
    path_save_dir.mkdir(parents=True, exist_ok=True)
    centroids = df.apply(
        lambda x: np.array([x['centroid_y'], x['centroid_x']]), axis=1
    )
    colors = df['lineage_id'].apply(lambda x: color_map.get(x))
    df = df.assign(centroid=centroids, color=colors)
    track_visualizer = TrackVisualizer(shape=shape)
    idx_last = str('inf')
    for idx, df_g in df.groupby('index_sequence'):
        if idx not in indices_todo:
            continue
        segments = []
        colors_segments = []
        for _, row in df_g.iterrows():
            indices_segs = [
                int(x) for x in row.in_list[1: -1].replace(' ', '').split(',')
                if x.isdigit()
            ]
            if len(indices_segs) > 1:
                raise NotImplementedError
            if len(indices_segs) > 0:
                try:
                    segments.append((
                        df.loc[indices_segs[0], 'centroid'],
                        row.centroid,
                    ))
                    colors_segments.append(color_map.get(row['lineage_id']))
                except:
                    import pdb; pdb.set_trace()
        path_tif = Path(path_save_dir, f'{idx:03d}_tracks.tiff')
        labels = [
            f'{l}:{t}' for l, t in zip(
                df_g['lineage_id'].tolist(), df_g['track_id'].tolist()
            )
        ]
        img = track_visualizer.render(
            centroids=df_g['centroid'].tolist(),
            segments=segments,
            colors_centroids=df_g['color'].tolist(),
            colors_segments=colors_segments,
            labels=labels,
        )
        save_tif(path_tif, img)
        indices_todo.remove(idx)
        idx_last = idx
    LOGGER.info(f'Creating blanks for frames: {list(indices_todo)}')
    # Create blank images for frames with no tracks
    while indices_todo:
        idx = indices_todo.pop()
        path_tif = Path(path_save_dir, f'{idx:03d}_tracks.tiff')
        if idx > idx_last:
            img = track_visualizer.render(centroids=[], segments=[])
        else:
            img = np.zeros(shape + (3,), dtype=np.uint8)
        save_tif(path_tif, img)


def render_centroid_groups(
        centroids_0: np.ndarray,
        centroids_1: np.ndarray,
        shape: Sequence[int],
) -> None:
    """Plot centroids from two groups."""
    plt.switch_backend('TkAgg')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(centroids_0[:, -1], centroids_0[:, -2], marker='.')
    for idx, cen in enumerate(centroids_0):
        ax.text(cen[-1], cen[-2], s=idx)
    ax.scatter(centroids_1[:, -1], centroids_1[:, -2], marker='.')
    for idx, cen in enumerate(centroids_1):
        ax.text(cen[-1], cen[-2], s=idx)
    ax.set_ylim([0, shape[0]])
    ax.set_xlim([0, shape[1]])
    ax.invert_yaxis()
    plt.show()
