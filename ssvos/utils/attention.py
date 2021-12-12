"""Implement some local attention and masked attention operation"""
from mindspore.common import Tensor
from mindspore import ops, dtype as mstype, numpy as np


def pytorch_topk(x:Tensor, k, dim=-1):
    topk_op = ops.TopK()
    if dim == -1:
        return topk_op(x, k)
    else:
        dims = x.ndim
        last_dim = dims-1
        # transpose dim with the last dim
        transpose_op = ops.Transpose()
        ori_dim_list = list(range(dims))
        transposed_dim_list = ori_dim_list
        transposed_dim_list[dim] = last_dim
        transposed_dim_list[last_dim] = dim
        x = transpose_op(x, transposed_dim_list)
        # perform topk
        values, indices = topk_op(x, k)
        # transpose back
        values = transpose_op(values, transposed_dim_list)
        indices = transpose_op(indices, transposed_dim_list)
        return values, indices


def spatial_neighbor(batches,
                     height,
                     width,
                     neighbor_range,
                     dtype=mstype.float32,
                     dim=1,
                     mode='circle') -> Tensor:
    # init some ops to be used
    assert dim in [1, 2]
    assert mode in ['circle', 'square']
    if mode == 'square':
        neighbor_range = (neighbor_range, neighbor_range)
        # [N, H, W, H, W]
        mask:Tensor = np.zeros(
            batches, height, width, height, width,dtype=dtype)
        for i in range(height):
            for j in range(width):
                top = max(0, i - neighbor_range[0] // 2)
                left = max(0, j - neighbor_range[1] // 2)
                bottom = min(height, i + neighbor_range[0] // 2 + 1)
                right = min(width, j + neighbor_range[1] // 2 + 1)
                mask[:, top:bottom, left:right, i, j] = 1

        mask = mask.view(batches, height * width, height * width)
        if dim == 2:
            mask = mask.transpose(0,2,1)
    else:
        radius = neighbor_range // 2
        grid_x, grid_y = np.meshgrid(
            np.arange(height, dtype=dtype),
            np.arange(width, dtype=dtype), indexing='ij')
        dist_mat = ((grid_x.view(height, width, 1, 1) -
                     grid_x.view(1, 1, height, width))**2 +
                    (grid_y.view(height, width, 1, 1) -
                     grid_y.view(1, 1, height, width))**2)**0.5
        mask = dist_mat < radius
        mask = mask.view(height * width, height * width)
    return mask.astype(mstype.bool_)


def masked_attention_efficient(query,
                               key,
                               value,
                               mask=None,
                               temperature=1,
                               topk=None,
                               normalize=True,
                               step=32,
                               non_mask_len=0) -> Tensor:
    """Performs local attention using a masked attention.

    Args:
        query (Tensor): Query tensor, shape (N, C, H, W)
        key (Tensor): Key tensor, shape (N, C, T, H, W)
        value (Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:
        new value.
    """
    # init some ops that will be used
    transpose_op = ops.Transpose()
    unsqueeze_op = ops.ExpandDims()
    create_zeros_tensor_op = ops.Zeros()
    create_ones_tensor_op = ops.Ones()
    # shape examination
    batches = query.shape[0]
    assert query.shape[0] == key.shape[0] == value.shape[0]
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = unsqueeze_op(key, 2)
        value = unsqueeze_op(value, 2)
    assert value.ndim == key.ndim == 5
    clip_len = key.shape[2]
    assert 0 <= non_mask_len < clip_len
    # normalize feature first
    if normalize:
        l2normalize_op = ops.L2Normalize(axis=1)
        query = l2normalize_op(query)
        key = l2normalize_op(key)
    # reshape q,k,v so that they can perform aatention easily
    att_channels, query_height, query_width = query.shape[1:]
    key_height, key_width = key.shape[3:]
    output_channels = value.shape[1]
    query_vec = query.view(batches, att_channels, query_height*query_width)
    key_vec = key.view(batches, att_channels, clip_len*key_height*key_width)
    value_vec = value.view(batches, att_channels,
                           clip_len*key_height*key_width) # [N, C, THW]
    output = create_zeros_tensor_op((batches, output_channels,
                                     query_height * query_width), query.dtype)
    # perform local attention step by step
    step = step or query_height * query_width
    for ptr in range(0, query_height*query_width, step):
        # compute affinity
        bmm_op = ops.BatchMatMul(transpose_a=True)
        step_affinity = bmm_op(
            key_vec, query_vec[..., ptr:ptr+step]) / temperature # [N, THW, step]
        if mask is not None:
            # get step mask
            if mask.ndim == 2:
                assert mask.shape == (key_height * key_width,
                                      query_height * query_width)
                step_mask = mask.view(1, 1, key_height * key_width,
                                      query_height*query_width)[..., ptr:ptr+step]
                expand_op = ops.BroadcastTo(
                    batches, clip_len - non_mask_len, -1, 1)
                step_mask = expand_op(step_mask)
                step_mask = step_mask.reshape(
                    (batches, -1, step_affinity.shape[2]))
            else:
                assert clip_len == 1
                assert non_mask_len == 0
                step_mask = mask[..., ptr:ptr + step]
            if non_mask_len > 0:
                cat_op = ops.Concat(axis=1)
                step_mask = cat_op([
                    create_ones_tensor_op((batches, non_mask_len * key_height * key_width,
                                           step_affinity.shape[2]), step_mask.dtype),
                    step_mask])
            step_affinity[~step_mask.astype(mstype.bool_)] = float('-inf')
        if topk is not None:
            topk_affinity, topk_indices = pytorch_topk(step_affinity, topk, dim=1) # [N, topk, step]
            # here we should use a standard index_select
            _value_vec = transpose_op(value_vec, (1,0,2)).reshape(output_channels, -1) # [C, NTHW]
            _tokp_indices = topk_indices.rehsape(-1) # [N*topk*step]
            topk_value = _value_vec[:, _tokp_indices] # [C, N*topk*step]
            topk_value.reshape(output_channels, *topk_indices.shape)
            topk_value = transpose_op(topk_value, (1,0,2,3)) # [N, C, topk, step]

            softmax_along_axis1_op = ops.Softmax(axis=1)
            topk_affinity = softmax_along_axis1_op(topk_affinity)

            topk_affinity = unsqueeze_op(topk_affinity, 1) # [N, 1, topk, step]
            step_output = (topk_affinity*topk_value).sum(2) # [N, C, step]
        output[...,ptr:ptr+step] = step_output
    
    output = output.reshape(batches, output_channels, query_height, query_width)

    return output # [N, C, H, W]
