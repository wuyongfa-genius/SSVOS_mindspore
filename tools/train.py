"""The train script in MindSpore to train my own model."""
import argparse
import os

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


def add_args():
    parser = argparse.ArgumentParser(
        description="MindSpore template train script")
    # enviornment args
    parser.add_argument('--device_target', type=str, default='GPU',
                        help='Device target, Currently GPU,Ascend are supported.')
    parser.add_argument('--distribute', type=bool, default=False,
                        help='Run distributed training.')
    parser.add_argument('--device_num', type=int,
                        default=1, help='Device num.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id, default is 0.')
    # dataset
    parser.add_argument('--dataset_root', type=str,
                        default='/data/DATASETS/Youtube_VOS/2018', help='dataset root path')
    parser.add_argument('--ann_file', type=str, default='ytvos_2018_raw_frames.txt',
                        help='path to annotation file')
    parser.add_argument('--num_frames', type=int, default=4, help='how many frames in a clip.')
    # work dir and log args
    parser.add_argument('--workdir', type=str, default='./exps', help='work dir in which stores\
                    logs and ckpts')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='How often to print log infos')
    parser.add_argument('--save_interval', type=int,
                        default=1, help='How often to save ckpts')
    # training args and hyper params
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num workers to load dataset.')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training, \
                    default is 100.')
    parser.add_argument('--resume_from', default=None, help='Resume training from a saved\
                    ckpt before.')
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

    # init your train dataloader here
    train_dataset = RawFrameDataset(args.dataset_root, args.ann_file, num_frames=args.num_frames)
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
                            weight_decay=1e-5)
    # init train net
    train_net = nn.TrainOneStepCell(model, optimizer)

    # load saved ckpt if we are resuming training or finetuning
    start_epoch = 0
    global_step = 0
    if args.resume_from is not None:
        ckpt = load_checkpoint(args.resume_from)
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
    ckpt_dir = os.path.join(args.workdir, 'ckpt')
    ckpt_cb = MyModelCheckpoint(ckpt_dir, interval=args.save_interval, rank=rank)
    log_dir = os.path.join(args.workdir, 'log')
    mindsight_cb = MindSightLoggerCallback(
        log_dir, log_interval=args.log_interval, rank=rank)
    console_log_cb = ConsoleLoggerCallBack(log_interval=args.log_interval)
    callbacks = [console_log_cb, mindsight_cb, ckpt_cb]
    # you can define a validation callback to validate

    # train network
    master_only_info("[INFO] Start training...")
    model.train(args.epoch_size, train_dataloader, callbacks=callbacks)
    master_only_info("[INFO] TRAINING DONE!")


if __name__ == "__main__":
    from mindspore.train import save_checkpoint,load_checkpoint, load_param_into_net
    from ssvos.models.backbones.resnet import CustomResNet
    # main()
    vtn = VideoTransformerNetwork(seqlength=8)
    byol = BYOL(encoder=vtn)
    criterion = CosineSimilarityLoss()
    model = NetWithSymmetricLoss(byol, criterion)
    lr = 0.1
    lr_scheduler = CosineDecayLRWithWarmup(lr, min_lr=1e-5, total_steps=10*100,
                                           warmup_steps=10*100)
    # init your optimizer here
    optimizer = nn.Momentum(model.trainable_params(), lr_scheduler, momentum=0.9,
                            weight_decay=1e-5)
    # init train net
    train_net = nn.TrainOneStepCell(model, optimizer)
    model = Model_with_start_states(train_net, amp_level='O0',
                                    start_epoch=0, start_step=0)
    save_checkpoint(model._train_network, 'full_model.ckpt')
    param_dict = load_checkpoint('full_model.ckpt')
    resnet = CustomResNet(depth=50)
    resnet_param = {}
    for k, v in param_dict.items():
        if k.startswith('online_encoder.0.frame_feat_extractor.'):
            k = k[38:]
            resnet_param.update(**{k:v})
    not_load_params = load_param_into_net(resnet, resnet_param)

    