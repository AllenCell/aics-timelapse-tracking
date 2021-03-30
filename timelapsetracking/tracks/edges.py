"""Functions to find connections between nodes in time-lapse graph."""

import logging

import pandas as pd

from timelapsetracking.tracks.correspondances import find_correspondances

from tqdm import tqdm

import numpy as np

from math import isnan


LOGGER = logging.getLogger(__name__)


def add_edges(
        df: pd.DataFrame,
        col_index_sequence: str = 'index_sequence',
        **kwargs,
) -> pd.DataFrame:
    """Add edges to track graph based on centroids.

    Parameters
    ----------
    df
        Input time-lapse graph with unconnected nodes.
    col_index_sequence
        DataFrame column corresponding to the time-lapse frame/sequence number.
    **kwargs
        Additional arguments to be passed to find_correspondances function.

    Returns
    -------
    pd.DataFrame
        Time-lapse graph with added edges

    """
    if col_index_sequence not in df.columns:
        raise ValueError(f'"{col_index_sequence}" column expected in df')
    df_prev = None
    cols_zyx = ['centroid_z', 'centroid_y', 'centroid_x']

    df_edges = (
        pd.DataFrame(index=df.index, columns=['in_list', 'out_list'])
        .fillna('[]')
    )

    df_edges['label_img'] = df['label_img']
    df_edges['index_sequence'] = df['index_sequence']
    df_edges['volume'] = df['volume']
    df_edges['edge_distance'] = df['edge_distance']
    df_edges['edge_cell'] = df['edge_cell']
    df_edges['is_pair'] = df['is_pair']
    df_edges['has_pair'] = df['has_pair']

    for col in cols_zyx:
        df_edges[col] = df[col]

    pairs_to_redo = []

    index_vals = df[col_index_sequence].unique()
    for idx_s in tqdm(range(index_vals.min()+1, index_vals.max() + 1)):

        LOGGER.info(f'Processing {col_index_sequence}: {idx_s}')
        print(f'Processing {col_index_sequence}: {idx_s}')
        df_prev = df.query(f'{col_index_sequence} == (@idx_s-1)')
        keep_idxs = [index for index, row in df_prev.iterrows() 
                        if not row['is_pair']]
        df_prev = df_prev.loc[keep_idxs].copy()

        df_curr = df.query(f'{col_index_sequence} == @idx_s')
        # keep_idxs = [index for index, row in df_curr.iterrows() 
        #                 if not row['has_pair']]
        # df_curr = df_curr.loc[keep_idxs].copy()


        
        has_pair = [idx for idx, row in df_prev.iterrows() 
                        if row['has_pair']]
        
        is_pair = [idx for idx, row in df_curr.iterrows() 
                        if row['is_pair']]

        centroids_prev = df_prev.filter(cols_zyx).values
        centroids_curr = df_curr.filter(cols_zyx).values
        volumes_prev, volumes_curr = None, None

        has_pair_curr = df_curr['has_pair'].values

        # import pdb; pdb.set_trace()
        if 'volume' in df_curr.columns:
            volumes_prev = df_prev['volume'].values#to_numpy()
            volumes_curr = df_curr['volume'].values#to_numpy()
        # edge_distances_prev = df_prev['edge_distance'].values
        # edge_distances_curr = df_curr['edge_distance'].values
        edges = find_correspondances(
            centroids_a=centroids_prev,
            centroids_b=centroids_curr,
            volumes_a=volumes_prev,
            volumes_b=volumes_curr,
            # cost_add=edge_distances_curr,
            # cost_delete=edge_distances_prev,
            has_pair_b=has_pair_curr,
            **kwargs
        )

        pair_edges = []

        for edge_from, edge_to in edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)

            idxs_curr = list(df_curr.index[edge_to])


            # skip edges pointing to pair nodes for next step
            skip = False
            for idx_curr in idxs_curr:
                if idx_curr in is_pair:
                    # print('edge to is pair')
                    # print((edge_from, edge_to))
                    pair_edges.append((edge_from, edge_to))
                    skip = True
                    break
            if skip:
                continue
             
            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            for idx_curr in idxs_curr:
                df_edges.loc[idx_curr, 'in_list'] = f'[{idx_prev}]'

        

        #redo edge mapping and splitting for edge nodes
        df_curr_has_pair = df.query(f'{col_index_sequence} == @idx_s')
        keep_idxs = [index for index, row in df_curr_has_pair.iterrows() 
                        if row['has_pair']]
        df_curr_has_pair = df_curr_has_pair.loc[keep_idxs].copy()

        for edge_from, edge_to in pair_edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)

            idxs_curr = df_curr.index[edge_to]

            if df_edges.loc[idxs_curr, 'is_pair'].values:
                child_list = []
                for label in df_edges.loc[idxs_curr,'label_img'].values[0]:
                    child_list.append(df_curr_has_pair.index[df_curr_has_pair['label_img'] == label].values[0])
                
                idxs_curr = child_list
                
            print('pair match')
            print(idx_prev)
            print(idxs_curr)
            # import pdb
            # pdb.set_trace()

            for idx_curr in idxs_curr:
                divorce_cell = eval(df_edges.loc[idx_curr, 'in_list'])
                if len(divorce_cell) > 0:
                    # print('overlapping match to pair cell. Overwriting previous match')
                    print('divorcing cell')
                    for cell in divorce_cell:
                        df_edges.loc[divorce_cell[0], 'out_list'] = '[]'

            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            df_edges.loc[idxs_curr, 'in_list'] = f'[{idx_prev}]'

        # find pair nodes that didn't match to attempt fix
        if is_pair is not None:
            matched_nodes = [df_curr.index[edge_to] for edge_from, edge_to in edges if not ((edge_from is None) or (edge_to is None))]
            # import pdb; pdb.set_trace()
            missed_pair_idxs = [pair_idx for pair_idx in is_pair if not pair_idx in matched_nodes]
            for pair in missed_pair_idxs:
                pairs_to_redo.append(pair)
        
    # redo pairs with no parent cell
    print(f'{len(pairs_to_redo)} pairs to redo')
    df_edges = add_pseudo_pairs(df_edges, pairs_to_redo, col_index_sequence, **kwargs)
    
    # attempt to bridge non-edge cells with missing graph edges
    df_edges = bridge_edges(df_edges, col_index_sequence, 3, **kwargs)
    

    # remove pair nodes in cuur from full dataframe
    remove_idx = [index for index, row in df_edges.iterrows() 
                    if row['is_pair']]
    
    df_edges = df_edges.drop(df_edges[df_edges.is_pair].index)


    df_edges = df_edges.drop(columns=['label_img', 
                                        'index_sequence', 
                                        'volume', 
                                        'edge_distance', 
                                        'edge_cell', 
                                        'is_pair',
                                        'has_pair'])
    df_edges = df_edges.drop(columns=cols_zyx)
    return df.join(df_edges)


def add_pseudo_pairs(
        df_edges: pd.DataFrame, 
        pairs_to_redo, 
        col_index_sequence: str = 'index_sequence',
        **kwargs
    ) -> pd.DataFrame:
    """Attempt to bridge gaps in node graph.

    Parameters
    ----------
    df_edges
        List of nodes after intial tracking with missing graph edges.
    col_index_sequence
        DataFrame column corresponding to the time-lapse frame/sequence number.
    window
        Number of timepoints in the future to look for potential edges.
    **kwargs
        Additional arguments to be passed to find_correspondances function.

    Returns
    -------
    pd.DataFrame
        Time-lapse graph with added edges

    """

    cols_zyx = ['centroid_z', 'centroid_y', 'centroid_x']

    index_vals = df_edges[col_index_sequence].unique()
    prev = None
    prev_c1 = None
    prev_c2 = None
    for pair_idx in tqdm(pairs_to_redo):

        pair_node = df_edges.loc[pair_idx]
        timepoint = pair_node[col_index_sequence]
        
        label_1 = pair_node['label_img'][0]
        label_2 = pair_node['label_img'][1]
        
        child_1 = df_edges.query(f'{col_index_sequence} == {timepoint} and label_img == {label_1}')
        child_2 = df_edges.query(f'{col_index_sequence} == {timepoint} and label_img == {label_2}')
        
        print(f'Starting in {timepoint}')
        pseudo_child_1, pseudo_child_2 = find_psuedo_pair(child_1, child_2, df_edges, col_index_sequence, start=True)
        
        pseudo_pair = {}
        if pseudo_child_1 is not None or pseudo_child_2 is not None:
            # import pdb; pdb.set_trace()
            print('Found cells for pseudo-pair')
            pseudo_pair['volume'] = ((pseudo_child_1['volume'].values[0] + pseudo_child_2['volume'].values[0]) + ((pseudo_child_1['volume'].values[0] + pseudo_child_2['volume'].values[0])/2))/2
            
            lbl_1 = pseudo_child_1['label_img']
            if isinstance(lbl_1, pd.Series):
                lbl_1 = lbl_1.values[0]
            lbl_2 = pseudo_child_2['label_img']
            if isinstance(lbl_2, pd.Series):
                lbl_2 = lbl_2.values[0]
            pair_labels = [lbl_1, lbl_2]

            pair_labels.sort()
            pseudo_pair['label_img'] = str(pair_labels)
            pseudo_pair['is_pair'] = True
            pseudo_pair['has_pair'] = False
            pseudo_pair['in_list'] = '[]'
            pseudo_pair['out_list'] = '[]'

            # try:
            pseudo_pair['edge_cell'] = bool(pseudo_child_1['edge_cell'].values[0] + pseudo_child_2['edge_cell'].values[0])
            # except:
            # pseudo_pair['edge_cell'] = bool(pseudo_child_1['edge_cell'] + pseudo_child_2['edge_cell'])

            for idx_d, dim in enumerate('zyx'):
                # if isinstance(pseudo_child_1,pd.DataFrame):
                pseudo_pair[f'centroid_{dim}'] = (pseudo_child_1[f'centroid_{dim}'].values[0] + pseudo_child_2[f'centroid_{dim}'].values[0])/2
                # else:
                # pseudo_pair[f'centroid_{dim}'] = (pseudo_child_1[f'centroid_{dim}'] + pseudo_child_2[f'centroid_{dim}'])/2
            
            pseudo_pair['edge_distance'] = (pseudo_child_1['edge_distance'].values[0] + pseudo_child_2['edge_distance'].values[0])/2
            pseudo_pair[col_index_sequence] = int(pseudo_child_1[col_index_sequence])

            pseudo_pair = pd.DataFrame(pseudo_pair, index=[df_edges.shape[0]+1])
            pseudo_pair.index.name = 'node_id'

            pseudo_pair.reindex(columns=df_edges.columns)

            df_edges = df_edges.append(pseudo_pair)
            prev = pseudo_pair
            
            in_1 = pseudo_child_1['in_list']
            in_2 = pseudo_child_2['in_list']
            # if isinstance(in_1,str):
            #     in_1 = eval(in_1)
            # elif isinstance(in_1,pd.Series):
            in_1 = eval(in_1.values[0])
            # if isinstance(in_2,str):
            #     in_2 = eval(in_2)
            # elif isinstance(in_2,pd.Series):
            in_2 = eval(in_2.values[0])
            
            if len(in_1)>0:
                try:
                    df_edges.loc[in_1, 'out_list'] = '[]'
                except:
                    import pdb; pdb.set_trace()
            if len(in_2)>0:
                df_edges.loc[in_2, 'out_list'] = '[]'
            
            # if isinstance(pseudo_child_1,pd.Series):
            #     df_edges.loc[pseudo_child_1.name, 'in_list'] = '[]'
            # else:
            df_edges.loc[pseudo_child_1.index, 'in_list'] = '[]'
            df_edges.loc[pseudo_child_1.index, 'has_pair'] = True
            # if isinstance(pseudo_child_2,pd.Series):
            #     df_edges.loc[pseudo_child_2.name, 'in_list'] = '[]'
            # else:
            df_edges.loc[pseudo_child_2.index, 'in_list'] = '[]'
            df_edges.loc[pseudo_child_2.index, 'has_pair'] = True

            print('psuedo-pair found and added')
            prev_c1 = pseudo_child_1
            prev_c2 = pseudo_child_2
        else:
            print('No new pair found')
        
    return df_edges

def find_psuedo_pair(child_1, child_2, df_edges, col_index_sequence, start=False):
    if child_1 is None or child_2 is None:
        print('Error, passed none down')
        return None, None
    try:
        if child_1[col_index_sequence].values[0] <= 0:
            print('reached to start of sequence')
            return None, None
    except:
        if child_1[col_index_sequence] <= 0:
            print('reached to start of sequence')
            return None, None

    
    in_node_1 = eval(child_1['in_list'].values[0])
    in_node_2 = eval(child_2['in_list'].values[0])
    out_node_1 = eval(child_1['out_list'].values[0])
    out_node_2 = eval(child_2['out_list'].values[0])
    
    time = child_1[col_index_sequence].values[0]

    #no pairs if there is split in linteage prior
    if len(out_node_1) > 1 or len(out_node_2) > 1:
        print(f'found earlier split at timepoint {time}')
        return None, None
    # return children if one or both have no upstream cell
    if (child_1.has_pair.values[0] or child_2.has_pair.values[0]) and not start:
        print('found an existing pair')
        return None, None
    elif len(in_node_1) == 0 or len(in_node_2) == 0:
        print(f'found pair at timepoint {time}')
        return child_1, child_2
    # otherwise recursively search further back
    else:
        # import pdb; pdb.set_trace()
        new_child_1 = df_edges.loc[[in_node_1[0]]]
        new_child_2 = df_edges.loc[[in_node_2[0]]]
        pseudo_child_1, pseudo_child_2 = find_psuedo_pair(new_child_1, new_child_2, df_edges, col_index_sequence)
        return pseudo_child_1, pseudo_child_2



def bridge_edges(
        df_edges: pd.DataFrame, 
        col_index_sequence: str = 'index_sequence',
        window: int = 3, 
        **kwargs
    ) -> pd.DataFrame:
    """Attempt to bridge gaps in node graph.

    Parameters
    ----------
    df_edges
        List of nodes after intial tracking with missing graph edges.
    col_index_sequence
        DataFrame column corresponding to the time-lapse frame/sequence number.
    window
        Number of timepoints in the future to look for potential edges.
    **kwargs
        Additional arguments to be passed to find_correspondances function.

    Returns
    -------
    pd.DataFrame
        Time-lapse graph with added edges

    """

    cols_zyx = ['centroid_z', 'centroid_y', 'centroid_x']

    index_vals = df_edges[col_index_sequence].unique()
    if window < 0:
        window = 0
    
    for idx_s in tqdm(range(index_vals.min()+1, index_vals.max() + 1)):

        LOGGER.info(f'Processing {col_index_sequence}: {idx_s}')
        df_prev = df_edges.query(f'{col_index_sequence} == (@idx_s-1)')
        keep_idxs = [idx for idx, row in df_prev.iterrows() 
                        if len(eval(row['out_list'])) == 0 
                        and not row['edge_cell']
                        and not row['is_pair']]
        df_prev = df_prev.loc[keep_idxs].copy()

        df_curr = df_edges.query(f'{col_index_sequence} >= @idx_s and {col_index_sequence} <= @idx_s + {window}')
        keep_idxs = [idx for idx, row in df_curr.iterrows() 
                        if len(eval(row['in_list'])) == 0
                        and not row['edge_cell']]
        df_curr = df_curr.loc[keep_idxs].copy()
        #redo edge mapping and splitting for edge nodes
        df_curr_has_pair = df_edges.query(f'{col_index_sequence} >= @idx_s and {col_index_sequence} <= @idx_s + {window}')
        df_curr_has_pair = df_curr_has_pair.loc[df_curr_has_pair.has_pair]

        if len(df_prev.index) == 0 or len(df_curr.index) == 0:
            continue
        
        # import pdb
        # pdb.set_trace()
        has_pair = [idx for idx, row in df_prev.iterrows() 
                        if row['has_pair']]
        is_pair = [idx for idx, row in df_curr.iterrows() 
                        if row['is_pair']]

        centroids_prev = df_prev.filter(cols_zyx).values
        centroids_curr = df_curr.filter(cols_zyx).values
        volumes_prev, volumes_curr = None, None
        
        has_pair_curr = df_curr['has_pair'].values

        # import pdb; pdb.set_trace()
        if 'volume' in df_curr.columns:
            volumes_prev = df_prev['volume'].values#to_numpy()
            volumes_curr = df_curr['volume'].values#to_numpy()
        # edge_distances_prev = df_prev['edge_distance'].values
        # edge_distances_curr = df_curr['edge_distance'].values
        edges = find_correspondances(
            centroids_a=centroids_prev,
            centroids_b=centroids_curr,
            volumes_a=volumes_prev,
            volumes_b=volumes_curr,
            # cost_add=edge_distances_curr,
            # cost_delete=edge_distances_prev,
            has_pair_b=has_pair_curr,
            is_bridge=True,
            **kwargs
        )

        pair_edges = []

        new_edges = [edge for edge in edges if all(edge)]
        print(f'{len(new_edges)} new edges bridged')

        for edge_from, edge_to in new_edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)

            idxs_curr = list(df_curr.index[edge_to])


            # skip edges pointing to pair nodes for next step
            skip = False
            for idx_curr in idxs_curr:
                if idx_curr in is_pair:
                    # print('edge to is pair')
                    # print((edge_from, edge_to))
                    pair_edges.append((edge_from, edge_to))
                    skip = True
                    break
            if skip:
                continue
             
            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            for idx_curr in idxs_curr:
                df_edges.loc[idx_curr, 'in_list'] = f'[{idx_prev}]'

        # #redo edge mapping and splitting for edge nodes
        # df_curr_has_pair = df_edges.query(f'{col_index_sequence} >= @idx_s and {col_index_sequence} <= @idx_s + {window}')
        # keep_idxs = [index for index, row in df_curr_has_pair.iterrows() 
        #                 if row['has_pair']]
        # df_curr_has_pair = df_curr_has_pair.loc[keep_idxs].copy()
        for edge_from, edge_to in pair_edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)

            # pairs = [index for index, row in df_curr.iterrows() 
            #             if row['is_pair']] 

            # df_curr_not_pair = df_curr.copy()
            # for idx in pairs:
            #     df_curr_not_pair['label_img'][idx] = []

            idxs_curr = df_curr.index[edge_to]
            

            if df_edges.loc[idxs_curr, 'is_pair'].values[0]:
                child_list = []
                labels = df_edges.loc[idxs_curr,'label_img'].values[0]
                if isinstance(labels,str):
                    labels = eval(labels)
                for label in labels:
                    try:
                        child_list.append(df_curr_has_pair.index[df_curr_has_pair['label_img'] == label].values[0])
                    except:
                        import pdb
                        pdb.set_trace()
                
                idxs_curr = child_list
                
            print('pair match')
            print(idx_prev)
            print(idxs_curr)

            # import pdb
            # pdb.set_trace()

            for idx_curr in idxs_curr:
                divorce_cell = eval(df_edges.loc[idx_curr, 'in_list'])
                if len(divorce_cell) > 0:
                    # print('overlapping match to pair cell. Overwriting previous match')
                    print('divorcing cell')
                    for cell in divorce_cell:
                        df_edges.loc[cell, 'out_list'] = '[]'

            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            df_edges.loc[idxs_curr, 'in_list'] = f'[{idx_prev}]'

            # import pdb; pdb.set_trace()

    return df_edges