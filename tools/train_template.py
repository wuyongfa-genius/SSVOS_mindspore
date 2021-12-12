"""The template train script in MindSpore."""
import argparse
import os

from mindspore import context, nn, set_seed
from mindspore.train import Model
# from mindspore.train.callback import (CheckpointConfig, LossMonitor,
#                                       ModelCheckpoint, TimeMonitor)
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from ssvos.datasets.utils import DataLoader
from ssvos.utils.callbacks import (ConsoleLoggerCallBack,
                                   MindSightLoggerCallback, MyModelCheckpoint)
from ssvos.utils.dist_utils import init_dist
from ssvos.utils.lr_schedule import CosineDecayLRWithWarmup


def add_args():
    parser = argparse.ArgumentParser(
        description="MindSpore template train script")
    # enviornment args
    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='Device target, Currently GPU,Ascend are supported.')
    parser.add_argument('--distribute', type=bool, default=True,
                        help='Run distributed training.')
    parser.add_argument('--device_num', type=int,
                        default=1, help='Device num.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id, default is 0.')
    # dataset
    parser.add_argument('--dataset_root', type=str,
                        default='/data', help='dataset root path')
    parser.add_argument('--ann_file', type=str, default='.',
                        help='path to annotation file')
    # work dir and log args
    parser.add_argument('--workdir', type=str, default='./exps', help='work dir in which stores\
                    logs and ckpts')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='How often to print log infos')
    parser.add_argument('--save_interval', type=int,
                        default=1, help='How often to save ckpts')
    # training args and hyper params
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size, default is 128.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers to load dataset.')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training, \
                    default is 100.')
    parser.add_argument('--optimizer', type=str, default='Momentum', help='Optimizer, Currently only\
                    Momentum is supported.')
    parser.add_argument('--base_lr', type=float,
                        default=0.1, help='base learning rate.')
    parser.add_argument('--lr_schedule', type=str,
                        default='cosine', help='Learning rate decay schedule')
    parser.add_argument('--weight_decay', type=float,
                        default=3e-4, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int,
                        default=15, help='warmup epochs.')

    return parser.parse_args()


def main():
    args = add_args()
    # set a global seed
    set_seed(2563)
    # set to graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    # init dist
    if args.device_target != "Ascend" and args.device_target != "GPU":
        raise ValueError("Unsupported device target.")
    if args.distribute:
        rank, group_size = init_dist()
    else:
        NotImplementedError

    # init your train dataloader here
    train_dataset = None
    train_dataloader = DataLoader(train_dataset, args.batch_size, args.num_workers,
                                  shuffle=True, drop_last=True)
    train_dataloader = train_dataloader.build_dataloader()
    # init your train dataloader here
    val_dataset = None
    val_dataloader = DataLoader(val_dataset, args.batch_size, args.num_workers,
                                shuffle=False)
    val_dataloader = val_dataloader.build_dataloader()

    # init your model here
    model: nn.Cell = None  # best wrap your model with your loss
    # init your lr scheduler here
    dataset_size = train_dataloader.get_dataset_size()
    lr = args.base_lr * group_size * args.batch_size / 256.
    lr_scheduler = CosineDecayLRWithWarmup(lr, min_lr=1e-5, total_steps=args.epoch_size*dataset_size,
                                           warmup_steps=args.warmup_epochs*dataset_size)
    # init your optimizer here
    optimizer = nn.Momentum(model.trainable_params(), lr_scheduler, momentum=0.9,
                            weight_decay=1e-5)
    # init train net
    train_net = nn.TrainOneStepCell(model, optimizer)
    # set amp_level to 'O2' to use fp16 with dynamic loss scale
    model = Model(train_net, amp_level='O1', boost_level='O1')
    # set callbacks
    ckpt_dir = os.path.join(args.workdir, 'ckpt')
    ckpt_cb = MyModelCheckpoint(ckpt_dir, interval=args.save_interval)
    log_dir = os.path.join(args.workdir, 'log')
    mindsight_cb = MindSightLoggerCallback(
        log_dir, log_interval=args.log_interval)
    console_log_cb = ConsoleLoggerCallBack(log_interval=args.log_interval)
    # train network
    model.train(args.epoch_size, train_dataloader, callbacks=[
                console_log_cb, mindsight_cb, ckpt_cb])


if __name__ == "__main__":
    main()
