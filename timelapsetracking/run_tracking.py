import argparse
import numpy as np
from glob import glob
import sys
import logging
from pathlib import Path
import traceback
import fire
import traceback

from . import dir_to_nodes
from .tracks.edges import add_edges
from .tracks import add_connectivity_labels
from .viz_utils import visualize_tracks_2d
from aicsimageio import AICSImage

######################################################################

# Perform Nucleus Tracking Algorithm/Workflow

######################################################################

log = logging.getLogger()
# Note: basicConfig should only be called in bin scripts (CLIs).
# https://docs.python.org/3/library/logging.html#logging.basicConfig
# "This function does nothing if the root logger already has handlers
# configured for it." As such, it should only be called once, and at
# the highest level (the CLIs in this case). It should NEVER be called
# in library code!
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s",
)

######################################################################


class Tracking(object):
    def __init__(self, folder):
        self.folder = folder

        # grad one image and determine the image shape
        filenames = glob(str(folder) + "/*.tiff")
        assert len(filenames) > 0, f"no tiff file found in {folder}"
        reader = AICSImage(filenames[0])
        self.img_shape = reader.shape[3:]  # STCZYX, only need ZYX dim

    def run(self, output_dir, edge_thresh_dist: int = 75, size_thresh: float = 0.9, debug: bool = False, visualize=False):

        # dir_to_nodes converts a folder of tif files in a dataframe
        # with the objects centroids
        df = dir_to_nodes.dir_to_nodes(self.folder, self.img_shape, num_processes=1)

        # add_edges implements the correspondance between objects accross time
        np.set_printoptions(threshold=sys.maxsize, precision=2, linewidth=250)
        df_edges = add_edges(df, thresh_dist=edge_thresh_dist, thresh_size = size_thresh)

        # TODO: add comment @Filip
        df_edges = add_connectivity_labels(df_edges)

        df_edges.to_csv(f"{output_dir}/edges.csv")

        # visualizae tracking results
        if visualize:
            visualize_tracks_2d(
                df=df_edges,
                shape=(self.img_shape[-2], self.img_shape[-1]),
                path_save_dir=Path(f"{output_dir}/visualization_2d/"),
            )


def main(
    input_dir, output, thresh_dist=75, size_thresh=0.9, visualize=False, debug=True
):
    try:
        # identify all the files to be processed
        input_dir = input_dir
        output_path = Path(output)

        # # prepare output path
        output_path.mkdir(exist_ok=True, parents=True)

        # This step is generic, but the parameter can be defined differently
        # for each instance of this step. To processing the timeplase movie
        # with name "XYZ", there will be an instance of this step called
        # "instance_seg_XYZ".
        edge_thresh_dist = thresh_dist

        exe = Tracking(input_dir)
        exe.run(output_path, edge_thresh_dist, size_thresh, debug, visualize)

    except Exception as e:
        log.error("=============================================")
        if debug:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)