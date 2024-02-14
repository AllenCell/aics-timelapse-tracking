# Timelapse Tracking

A repositroy for performing tracking of object tracking on timelapses of segmented, nucleus-tagged cell colonies. This repo is used as part of the Nuclear Morphogenesis [data generating pipeline](https://github.com/aics-int/morflowgenesis)

---
# Setup
In terminal, move to the directory which you want to keep a copy of this repo. Then follow the commands bellow to clone and install the repo into a fresh conda environment.

```bash
# clone repository
git clone git@github.com:aics-int/aics-timelapse-tracking.git

# create and activate conda env
conda create -n aics-tracking python==3.8
conda activate aics-tracking

#install repo
cd aics-timelapse-tracking
pip install -e .

```

---
# How to Use

**Note:** The methods and their default values are optimized for use on 20x fov's which have been upsampled to a 100x resolution. 

## Input Data Formatting
If you are running tracking on data generated using the instance segmentation from the NucMorph workflow then the input directory is correctly formatted. However, if you are using data from a different source then you must make sure the data is formatted in the following ways.

1) Every segmented frame of the timelapse is saved as a seperate, individual `.tiff` or `.tif` (not `.ome.tiff`) file within the same directory. They should be named such that sorting them alphabetically orders them correctly.
2) The images must be instance segmentations with a unique integer label for each segmented nucleus (this doesn't need to be unique across all of the frames)
3) Each image should be accompanied by a `.csv` file named `[image file name]_region_props.csv`. E.x. a image named `T_01.tiff` would be accompanied by `T_01_region_props.csv`
4) Each csv file must contain the following columns for each segmented nucleus

| Column Name | Data Type | Description |
| --- | --- | --- |
| CellLabel | int | the integer label in the segmentation |
| Volume | int/float | total volume of the nucleus, physical volume or voxel count |
| Edge_Cell | boolean | true if the segmentation is touching the edge of the fov |
| Pair | int | (Optional) the integer label of a nuclei determined to be its sibling from a recent mitotis (within the last few frames) |

## Running Cell Tracking

The tracking algorithm can be run programmatically in python as follows
```python
# function imports
import pandas as pd
import numpy as np
from pathlib import Path

from timelapsetracking import dir_to_nodes
from timelapsetracking.tracks.edges import add_edges
from timelapsetracking.tracks import add_connectivity_labels
from timelapsetracking.viz_utils import visualize_tracks_2d

# define variables
input_dir = '' # folder containing input data
output_dir = '' # folder to output tracking results
image_shape = (110, 3121, 4621) # pixel dimensions of fov's (z,y,x)
distance_threshold = 75 # maximum euclidean distance (in pixels) for searching for potential matches during tracking
size_threshold = 0.9 # minimum difference in volume between two cell before algorithm is biased against matching them 

# read input directory
df = dir_to_nodes.dir_to_nodes(self.folder, self.img_shape, num_processes=8)

# run tracking algorithm
df_edges = add_edges(df, thresh_dist=distance_threshold, thresh_size = size_threshold)
df_edges = add_connectivity_labels(df_edges)

# save tracking output
df_edges.to_csv(f"{output_dir}/edges.csv")

# (Optional)
# visualize tracking results
visualize_tracks_2d(
    df=df_edges,
    shape=(self.img_shape[-2], self.img_shape[-1]),
    path_save_dir=Path(f"{output_dir}/visualization_2d/"),
)
```

The algorithm can also be run from the command line as follows:

```bash
# activate conda environment and navigate to repo directory
activate aics-tracking
cd [path/to/repo]
cd aics-timelapse-tracking/timelapsetracking

# run code
python run_tracking.py \\
    --input_dir [path to input data folder] \\
    --output [path to output folder] \\
    --thresh_dist [distance threshold] \\
    --size_thresh [size threshold] \\
    --visualize True #(Optional) include if you want tracking visualization to be created, can be left out otherwise

```

## Output Data

### CSV File
Tracking results are outputed to the output directory as a `.csv` file. If run using the command-line method this file is named `edges.csv`. The file will contain the following data for every segmentation across the timelapse data.

| Column Name | Data Type | Description |
| --- | --- | --- |
| node_id | int | row index |
| volume | int/float | nucleus volume, same as input data |
| label_img | int | the integer segmentation label, same as input data |
| is_pair | boolean | whether the nucleus is part of a mitotic pair from the input data and is treated the timepoint of the nuclear division as determined by the tracking algorithm |
| has_pair | boolean | if at any timepoint of the track `is_pair` is true |
| centroid_z | float | z coordinate of the segmented object's centroid |
| centroid_y | float | y coordinate of the segmented object's centroid |
| centroid_x | float | x coordinate of the segmented object's centroid |
| edge_distance | float | shortest euclidean distance from object's centroid to the edge of the fov |
| edge_cell | boolean | whether object is touching fov boundary, same as input data |
| index_sequence | int | frame number of segmented object |
| time_index | int | frame number of segmented object |
| path_tif | string | file path to segmented image |
| in_list | list(int) | `node_id` of previous instance of nucleus (e.x. `[4]`). Value is `[]` if track starts in this frame |
| out_list | list(int) | `node_id` of next instance of nucleus. If the nucleus underwent mitosis between frames, the list will contain the `node_id` values of its daughter cells in the next frame (e.x. `[20,50]`). Value is `[]` if track ends in this frame |
| lineage_id | int | unique label for every set of parent-daughter nuclei |
| track_id | int | unique label for every individually tracked nucleus, not including daughter cells |

### Visualization

If visualization is enabled with either method an additional subdirectory will be formed in the output location. This will save a tiff image for each frame which shows the centroid of each image, its `lineage_id` and `track_id` (in that order), and a trail of the tracked nuclei's recent positions.

# Development workflows
## Pre-commit: autoformatting
Create a virtual environment and install dependencies as normal.
Set up autoformatting with pre-commit. This will run `isort`, `black`, and `autoflake` locally on every commit.
```bash
pip install .[dev]
pre-commit install
```

## Test
Create a virtual environment and install dependencies as normal.
Install test dependencies and run the tests as follows.
```bash
pip install .[dev]
pytest
```

## Publish
1. Update the version number in `pyproject.toml` as part of a PR.
2. Merge the PR into main.
3. Tag the commit (replace v0.1.0 with your version).
```bash
git checkout main
git pull
git tag -a -m "v0.1.0" v0.1.0
```
4. Push your tag to trigger the PyPi release.
```bash
git push --follow-tags
```
5. [Check here](https://github.com/aics-int/aics-timelapse-tracking/actions) that the Publish action succeeded.

Note: publishing to PyPi is not yet enabled for this package.

# Legal documents

- [License](LICENSE.txt) _(Before you release the project publicly please check with your manager to ensure the license is appropriate and has been run through legal review as necessary.)_
- [Contribution Agreement](CONTRIBUTING.md) _(Additionally update the contribution agreement to reflect
the level of contribution you are expecting, and the level of support you intend to provide.)_
