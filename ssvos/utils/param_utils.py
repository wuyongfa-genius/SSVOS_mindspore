"""Implement some model parameter utils."""
import math
from functools import reduce
from mindspore import nn
from mindspore import Parameter
from mindspore.common import initializer as init
from typing import Tuple


def _calculate_in_and_out(arr):
    dim = len(arr.shape)
    if dim < 2:
        raise ValueError("If initialize data with xavier uniform, "
                         "the dimension of data must greater than 1.")

    n_in = arr.shape[1]
    n_out = arr.shape[0]

    if dim > 2:
        counter = reduce(lambda x, y: x * y, arr.shape[2:])
        n_in *= counter
        n_out *= counter
    return n_in, n_out


def init_param(param: Parameter, Init: Tuple[init.Initializer, str], **init_kwargs):
    param_shape = param.shape
    param_dtype = param.dtype
    init_ = Init if isinstance(Init, str) else Init(**init_kwargs)
    param.set_data(init.initializer(init_, param_shape, param_dtype))


def default_weight_init(model: nn.Cell):
    """Default initialization for convnets with ReLU activations."""
    for name, cell in model.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv3d)):
            init_param(cell.weight, init.HeNormal,
                       mode='fan_out', nonlinearity='relu')
            if cell.bias is not None:
                init_param(cell.bias, 'zeros')
        elif isinstance(cell, nn.Dense):
            init_param(cell.weight, init.HeUniform,
                       negative_slope=math.sqrt(5))
            if cell.bias is not None:
                fan_in, _ = _calculate_in_and_out(cell.weight)
                bound = 1 / math.sqrt(fan_in)
                init_param(cell.bias, init.Uniform, scale=bound)
        elif isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d)):
            pass  # usually we use the default init in MindSpore


def divide_decay_no_decay_params(model: nn.Cell):
    decay_params = []
    no_decay_params = []
    for param in model.trainable_params():
        name = param.name
        if name.endswith('.bias') or name.endswith('.gamma') or name.endswith('.beta'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]
