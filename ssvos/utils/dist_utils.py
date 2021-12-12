"""Some distribute training utils."""
import os
import mindspore
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import communication as dist


def init_dist(parallel_mode=ParallelMode.DATA_PARALLEL):
    device_id = int(os.getenv('DEVICE_ID'))
    rank_size = int(os.getenv('RANK_SIZE'))
    context.set_context(device_id=device_id)

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                    gradients_mean=True,
                                    device_num=rank_size)
    dist.init()

    return dist.get_rank(), dist.get_group_size()