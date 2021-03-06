"""Train a MindSpore Model in ModelArts."""
import argparse
import numpy as np
import os
import moxing as mox

from mindspore import context, nn, set_seed, Model
from mindspore.nn.learning_rate_schedule import CosineDecayLR
from ssvos.datasets.utils import DataLoader
from ssvos.utils.callbacks import (ConsoleLoggerCallBack,
                                   MindSightLoggerCallback, MyModelCheckpoint)
from ssvos.utils.dist_utils import init_dist
from ssvos.utils.lr_schedule import CosineDecayLRWithWarmup
from ssvos.datasets import RawFrameDataset
from ssvos.models.BYOL import BYOL
from ssvos.models.backbones import VideoTransformerNetwork
from ssvos.utils.module_utils import NetWithSymmetricLoss
from ssvos.utils.loss_utils import CosineSimilarityLoss
from ssvos.utils.log_utils import master_only_info, set_logger_level_to


def add_args():
    parser = argparse.ArgumentParser(
        description="MindSpore ModelArts train script")
    # dataset
    parser.add_argument('--data_url', type=str,
                        required=True, help='dataset root path in obs')
    parser.add_argument('--ann_file', type=str, default='ytvos_2018_raw_frames.txt',
                        help='path wrt to data_url to annotation file')
    parser.add_argument('--num_frames', type=int, default=8, help='how many frames in a clip.')
    # work dir and log args
    parser.add_argument('--train_url', type=str, required=True, help='work dir in which stores\
                    logs and ckpts, physically in obs')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='How often to print log infos')
    parser.add_argument('--save_interval', type=int,
                        default=5, help='How often to save ckpts')
    # training args and hyper params
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers to load dataset.')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training, \
                    default is 100.')
    parser.add_argument('--optimizer', type=str, default='Momentum', help='Optimizer, Currently only\
                    Momentum is supported.')
    parser.add_argument('--base_lr', type=float,
                        default=0.2, help='base learning rate.')
    parser.add_argument('--lr_schedule', type=str,
                        default='cosine', help='Learning rate decay schedule')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int,
                        default=10, help='warmup epochs.')

    return parser.parse_args()


def main():
    NUM_HOSTS = 3
    MODELARTS_DATA_DIR = '/cache/dataset'
    MODELARTS_WORK_DIR = '/cache/output'
    args = add_args()
    set_logger_level_to()
    # set a global seed
    np.random.seed(2563)
    set_seed(2563)
    # set to graph mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    # init dist
    rank, group_size = init_dist()

    MODELARTS_DATA_DIR = os.path.join(MODELARTS_DATA_DIR, f'_{rank}')
    MODELARTS_WORK_DIR = os.path.join(MODELARTS_WORK_DIR, f'_{rank}')
    os.makedirs(MODELARTS_WORK_DIR, exist_ok=True)
    ## init your train dataloader here
    # download dataset from obs to cache if train on ModelArts
    master_only_info('[INFO] Copying dataset from obs to ModelArts...', rank=rank)
    mox.file.copy_parallel(src_url=args.data_url, dst_url=MODELARTS_DATA_DIR)
    master_only_info('[INFO] Done. Start training...', rank=rank)
    train_dataset = RawFrameDataset(MODELARTS_DATA_DIR, args.ann_file, args.num_frames)
    train_dataloader = DataLoader(train_dataset, args.batch_size, args.num_workers,
                                  shuffle=True, drop_last=True, distributed=True)
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
    lr_scheduler = CosineDecayLRWithWarmup(lr, min_lr=1e-5, total_steps=args.epoch_size*dataset_size*NUM_HOSTS,
                                           warmup_steps=args.warmup_epochs*dataset_size*NUM_HOSTS)
    # lr_scheduler = CosineDecayLR(min_lr=1e-5, max_lr=lr, decay_steps=args.epoch_size*dataset_size)
    # init your optimizer here
    optimizer = nn.Momentum(model.trainable_params(), lr_scheduler, momentum=0.9,
                            weight_decay=args.weight_decay)
    # init train net
    train_net = nn.TrainOneStepCell(model, optimizer)
    model = Model(train_net)

    # init callbacks
    ckpt_dir = os.path.join(MODELARTS_WORK_DIR, 'ckpts')
    ckpt_cb = MyModelCheckpoint(ckpt_dir, interval=args.save_interval, rank=rank)
    log_dir = os.path.join(MODELARTS_WORK_DIR, 'logs')
    mindsight_cb = MindSightLoggerCallback(
        log_dir, log_interval=args.log_interval, rank=rank)
    console_log_cb = ConsoleLoggerCallBack(log_interval=args.log_interval, rank=rank)
    callbacks = [console_log_cb, mindsight_cb, ckpt_cb]
    # you can define a validation callback to validate

    # train network
    master_only_info("[INFO] Start training...", rank=rank)
    model.train(args.epoch_size, train_dataloader, callbacks=callbacks)
    master_only_info("[INFO] TRAINING DONE!", rank=rank)

    # upload ckpts and logs from ModelArts to obs
    master_only_info("[INFO] Copying workdir contents from ModelArts to OBS...", rank=rank)
    mox.file.copy_parallel(
                src_url=MODELARTS_WORK_DIR, dst_url=args.train_url)
    master_only_info("[INFO] Done.", rank=rank)


if __name__ == "__main__":
    main()
