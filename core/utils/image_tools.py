import numpy as np


def max_count_pool(arr):
    val, idx = np.unique(arr, return_counts=True)
    return val[np.argsort(idx)[-1]]


# completion for items by max pooling
def complete2d(src_map, loops, keep_margin=False):
    """completion for 2d image (repeat for multi-channel objects)"""
    for l in range(loops):
        if keep_margin:
            poolfunc = np.max if l < 1 else max_count_pool
        else:
            poolfunc = np.max
        comp_map = pooling2d(src_map, 1, 1, poolfunc)  # 3x3 operation
        # comp_map = fastcompletion(src_map)
        invalid_idx = (src_map == -1000)  # invalid condition
        # print(np.sum(invalid_idx) / (src_map.shape[0]**2), comp_map.min())
        src_map = comp_map * invalid_idx + src_map * (1 - invalid_idx)
    # edge completion
    src_map = complete_edge(src_map)
    return src_map


# complete images on the edges
def complete_edge(src_map):
    out_map = src_map.copy()
    # edge
    out_map[0, :] = src_map[1, :]
    out_map[-1, :] = src_map[-2, :]
    out_map[:, 0] = src_map[:, 1]
    out_map[:, -1] = src_map[:, -2]
    # corner
    out_map[0, 0] = src_map[1, 1]
    out_map[0, -1] = src_map[1, -2]
    out_map[-1, 0] = src_map[-2, 1]
    out_map[-1, -1] = src_map[-2, -2]
    # merge source map
    invalid_idx = (src_map < -100)
    out_map = out_map * invalid_idx + src_map * (1 - invalid_idx)
    return out_map


def pooling2d(inputMap, poolSize, poolStride, pool_func):
    """Completion by pooling on sliding windows for 2d array
    Args:
        inputMap: input array of the pooling layer
        poolSize: X-size(equivalent to Y-size) of receptive field
        poolStride: the stride size between successive pooling squares
        pool_func: pooling methods: np.max, np.mean, max_count_pool

    Returns:
        outputMap: output array of the pooling layer
    """
    # inputMap sizes
    in_row, in_col = np.shape(inputMap)

    # outputMap sizes
    out_row, out_col = int(np.floor(in_row / poolStride)), int(np.floor(in_col / poolStride))
    row_remainder, col_remainder = np.mod(in_row, poolStride), np.mod(in_col, poolStride)
    if row_remainder != 0:
        out_row += 1
    if col_remainder != 0:
        out_col += 1
    outputMap = np.zeros((out_row, out_col))

    # padding
    temp_map = np.lib.pad(inputMap, ((0, poolSize - row_remainder), (0, poolSize - col_remainder)), 'edge')

    for r_idx in range(poolSize, out_row - poolSize):
        for c_idx in range(poolSize, out_col - poolSize):
            startX = c_idx * poolStride
            startY = r_idx * poolStride
            poolField = temp_map[startY - poolSize:startY + poolSize + 1, startX - poolSize:startX + poolSize + 1]
            poolOut = pool_func(poolField)
            outputMap[r_idx, c_idx] = poolOut

    return outputMap
