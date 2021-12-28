"""Train a MindSpore Model in ModelArts."""
import argparse
import numpy as np
import os
import moxing as mox

from mindspore import context, nn, set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from ssvos.datasets.utils import DataLoader
from ssvos.utils.callbacks import (ConsoleLoggerCallBack,
                                   MindSightLoggerCallback, MyModelCheckpoint)
from ssvos.utils.dist_utils import init_dist
from ssvos.utils.lr_schedule import CosineDecayLRWithWarmup
from ssvos.utils.Model_wrapper import Model_with_start_states
from ssvos.datasets import RawFrameDataset
from ssvos.models.BYOL import BYOL
from ssvos.models.backbones import VideoTransformerNetwork
from ssvos.utils.module_utils import NetWithSymmetricLoss
from ssvos.utils.loss_utils import CosineSimilarityLoss
from ssvos.utils.log_utils import master_only_info, set_logger_level_to

MODELARTS_DATA_DIR = '/cache/dataset'
MODELARTS_PRETRAINED_DIR = '/cache/pretrained'
MODELARTS_WORK_DIR = '/cache/output'

def add_args():
    parser = argparse.ArgumentParser(
        description="MindSpore ModelArts train script")
    # enviornment args
    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='Device target, Currently GPU, Ascend are supported.')
    parser.add_argument('--distribute', type=bool, default=True,
                        help='Run distributed training.')
    # dataset
    parser.add_argument('--data_url', type=str,
                        required=True, help='dataset root path in obs')
    parser.add_argument('--ann_file', type=str, default='ytvos_2018_raw_frames.txt',
                        help='path wrt to data_url to annotation file')
    parser.add_argument('--num_frames', type=int, default=4, help='how many frames in a clip.')
    # work dir and log args
    parser.add_argument('--train_url', type=str, required=True, help='work dir in which stores\
                    logs and ckpts, physically in obs')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='How often to print log infos')
    parser.add_argument('--save_interval', type=int,
                        default=1, help='How often to save ckpts')
    # training args and hyper params
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size.')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='num workers to load dataset.')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training, \
                    default is 100.')
    parser.add_argument('--resume_from', default=None, help='Resume training from a saved\
                    ckpt before, the path is wrt the workdir.')
    parser.add_argument('--optimizer', type=str, default='Momentum', help='Optimizer, Currently only\
                    Momentum is supported.')
    parser.add_argument('--base_lr', type=float,
                        default=0.2, help='base learning rate.')
    parser.add_argument('--lr_schedule', type=str,
                        default='cosine', help='Learning rate decay schedule')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int,
                        default=5, help='warmup epochs.')

    return parser.parse_args()


def main():
    args = add_args()
    set_logger_level_to()
    # set a global seed
    np.random.seed(2563)
    set_seed(2563)
    # set to graph mode
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=args.device_target)
    # init dist
    if args.device_target != "Ascend" and args.device_target != "GPU":
        raise ValueError("Unsupported device target.")
    if args.distribute:
        rank, group_size = init_dist()
    else:
        rank, group_size = 0, 1

    ## init your train dataloader here
    # download dataset from obs to cache if train on ModelArts
    master_only_info('[INFO] Copying dataset from obs to ModelArts...', rank=rank)
    if rank == 0:
        mox.file.copy_parallel(src_url=args.data_url, dst_url=MODELARTS_DATA_DIR)
    master_only_info('[INFO] Done. Start training...', rank=rank)
    train_dataset = RawFrameDataset(MODELARTS_DATA_DIR, args.ann_file, args.num_frames)
    train_dataloader = DataLoader(train_dataset, args.batch_size, args.num_workers,
                                  shuffle=True, drop_last=True, distributed=args.distribute)
    train_dataloader = train_dataloader.build_dataloader()
    master_only_info("[INFO] Dataset loaded!", rank=rank)

    # init your model here
    vtn = VideoTransformerNetwork(seqlength=args.num_frames)
    byol = BYOL(encoder=vtn)
    criterion = CosineSimilarityLoss()
    model = NetWithSymmetricLoss(byol, criterion)
    master_only_info("[INFO] Model initialized!", rank=rank)
    # init your lr scheduler here
    dataset_size = train_dataloader.get_dataset_size()
    lr = args.base_lr * group_size * args.batch_size / 256.
    lr_scheduler = CosineDecayLRWithWarmup(lr, min_lr=1e-5, total_steps=args.epoch_size*dataset_size,
                                           warmup_steps=args.warmup_epochs*dataset_size)
    # init your optimizer here
    optimizer = nn.Momentum(model.trainable_params(), lr_scheduler, momentum=0.9,
                            weight_decay=args.weight_decay)
    # init train net
    train_net = nn.TrainOneStepCell(model, optimizer)

    # load saved ckpt if we are resuming training or finetuning
    start_epoch = 0
    global_step = 0
    if args.resume_from is not None:
        master_only_info('[INFO] Copying saved ckpts from OBS to ModelArts...', rank=rank)
        if rank == 0:
            mox.file.copy_parallel(src_url=os.path.join(args.train_url, 'ckpts', args.resume_from), dst_url=MODELARTS_PRETRAINED_DIR)
        master_only_info('[INFO] Done.', rank=rank)
        ckpt = load_checkpoint(os.path.join(MODELARTS_PRETRAINED_DIR, args.resume_from))
        if 'epoch' in ckpt.keys():
            start_epoch = ckpt['epoch']
        if 'global_step' in ckpt.keys():
            global_step = ckpt['global_step']
        # load model params and optimizer params
        load_param_into_net(train_net, ckpt)
        master_only_info("[INFO] Checkpoint loaded!", rank=rank)
    # set amp_level to 'O2' to use fp16 with dynamic loss scale
    model = Model_with_start_states(train_net, amp_level='O0',
                                    start_epoch=start_epoch, start_step=global_step)

    # init callbacks
    ckpt_dir = os.path.join(MODELARTS_WORK_DIR, 'ckpts')
    ckpt_cb = MyModelCheckpoint(ckpt_dir, interval=args.save_interval)
    log_dir = os.path.join(args.workdir, 'logs')
    mindsight_cb = MindSightLoggerCallback(
        log_dir, log_interval=args.log_interval)
    console_log_cb = ConsoleLoggerCallBack(log_interval=args.log_interval)
    callbacks = [console_log_cb, mindsight_cb, ckpt_cb]
    # you can define a validation callback to validate

    # train network
    master_only_info("[INFO] Start training...")
    model.train(args.epoch_size, train_dataloader, callbacks=callbacks)
    master_only_info("[INFO] TRAINING DONE!")

    # upload ckpts and logs from ModelArts to obs
    master_only_info("[INFO] Copying workdir contents from ModelArts to OBS...", rank=rank)
    if rank == 0:
        mox.file.copy_parallel(
                    src_url=MODELARTS_WORK_DIR, dst_url=args.train_url)
    master_only_info("[INFO] Done.", rank=rank)


if __name__ == "__main__":
    main()
