import numpy as np
import numpy.testing as npt
import pandas as pd

from timelapsetracking.tracks.analysis import add_delta_pos


def test_add_delta_pos():
    """Creates test time-lapse graph and checks delta position calculations."""
    df_i = pd.DataFrame(
        {
            'node_id': range(9),
            'in_list': [None, 0, 1, None, 3, 4, None, 6, None],
            'centroid_x': [1, 2, 3.1, 4, 6, 8.2, 9, 12, 666],
            'centroid_y': [2, 2, 2, 3, 3, 3, -3, -3, 777],
            'centroid_z': [0, -1, -3, 0, 3, 7, 0, -5, 888],
        }
    ).set_index('node_id')
    df_o = add_delta_pos(df_i)
    delta_x_exp = [np.nan, 1.0, 1.1, np.nan, 2.0, 2.2, np.nan, 3.0, np.nan]
    delta_y_exp = [np.nan, 0.0, 0.0, np.nan, 0.0, 0.0, np.nan, 0.0, np.nan]
    delta_z_exp = [np.nan, -1, -2, np.nan, 3, 4, np.nan, -5, np.nan]
    npt.assert_array_almost_equal(df_o['delta_centroid_x'], delta_x_exp)
    npt.assert_array_almost_equal(df_o['delta_centroid_y'], delta_y_exp)
    npt.assert_array_almost_equal(df_o['delta_centroid_z'], delta_z_exp)

    # Test case where not all positions are available
    df_i = df_i.filter(['in_list', 'centroid_z'])
    df_o = add_delta_pos(df_i)
    npt.assert_array_almost_equal(df_o['delta_centroid_z'], delta_z_exp)
