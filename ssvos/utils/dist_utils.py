"""Some distribute training utils."""
import os

from mindspore import context
from mindspore.context import ParallelMode
from mindspore import communication as dist


def init_dist(parallel_mode=ParallelMode.DATA_PARALLEL):
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_size = int(os.getenv('RANK_SIZE', '1'))
    context.set_context(device_id=device_id)

    context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                    gradients_mean=True,
                                    device_num=rank_size)
    dist.init()

    return dist.get_rank(), dist.get_group_size()


# if __name__=="__main__":
#     rank, group_size = init_dist()
#     print(rank)
#     print(group_size)