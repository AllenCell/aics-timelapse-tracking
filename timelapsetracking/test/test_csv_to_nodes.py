import numpy as np
import pandas as pd

from timelapsetracking.csv_to_nodes import csv_to_nodes


def test_edge_cell():
    """Tests node generation from a directory of dummy images."""
    meta_dict = {
        "time_index": "Timepoint",
        "index_sequence": "Timepoint",
        "volume": "Volume",
        "label_img": "CellLabel",
        "zyx_cols": ["Centroid_z", "Centroid_y", "Centroid_x"],
        "edge_cell": "Edge_Cell",
    }
    data = pd.DataFrame(
        {
            "Timepoint": [1, 1, 2, 2],
            "Volume": [1, 2, 3, 4],
            "CellLabel": [1, 2, 3, 4],
            "Centroid_z": [1, 2, 3, 4],
            "Centroid_y": [1, 2, 3, 4],
            "Centroid_x": [1, 2, 3, 4],
            "Edge_Cell": [False, True, False, True],
        }
    )
    df = csv_to_nodes(data, meta_dict, np.array([5, 5, 5]))
    assert df["edge_cell"].tolist() == [False, True, False, True]
