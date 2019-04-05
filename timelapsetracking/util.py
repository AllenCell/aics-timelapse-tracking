"""General utility functions."""


from typing import Callable, List, Sequence
import os
import time


def report_run_time(fn: Callable):
    """Decorator to report function execution time."""
    def wrapper(*args, **kwargs):
        tic = time.time()
        retval = fn(*args, **kwargs)
        toc = time.time()
        name = fn.__module__ + '.' + fn.__qualname__
        print(f'{name} executed in {toc - tic:.2f} s.')
        return retval
    return wrapper


def images_from_dir(
        path_dir: str,
        extensions: Sequence[str] = ('.tif', '.tiff'),
) -> List[str]:
    """Returns a sorted list of images found in the input directory.

    Parameters
    ----------
    path_dir
        Path to input directory.
    extensions
        File extensions to identify a file as an image.

    Returns
    -------
    List[str]
        Sorted list of paths to images in input directory.

    """
    paths_img = sorted(
        [
            p.path for p in os.scandir(path_dir)
            if any(ext in p.path.lower() for ext in extensions)
        ]
    )
    return paths_img
