"""The script to evaluate VOS on ModelArts. You need to create a result dir on OBS and copy your 
pretrained weight under it before you launch the script on ModelArts."""
import argparse
import os

import numpy as np
from PIL import Image
from functools import partial
from mindspore import nn, numpy as msnp, ops, context
from ssvos.datasets import (DAVIS_VAL, imwrite_indexed, norm_mask, default_palette,
                            DataLoader)
from ssvos.models.backbones import CustomResNet
from ssvos.utils.dist_utils import init_dist
from ssvos.utils.attention import spatial_neighbor, masked_attention_efficient
from ssvos.utils.log_utils import master_only_info, set_logger_level_to
from ssvos.utils.ckpt_utils import load_partial_param_into_net
import moxing as mox


def add_args():
    parser = argparse.ArgumentParser(
        'Evaluation with video object segmentation on DAVIS 2017')
    parser.add_argument('--distribute', type=bool, default=True,
                        help='Run distributed testing.')
    parser.add_argument('--pretrained_weight', default='',
                        type=str, help="Path to pretrained weight to evaluate. The path is wrt the OBS result path `train_url`.")
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['deit_tiny', 'deit_small', 'resnet18', 'resnet50'], help='Architectures.')
    parser.add_argument('--out_stride', default=8, type=int,
                        help='Output stride of the model.')
    parser.add_argument("--encoder_key", default="online_encoder.0.frame_feat_extractor.",
                        type=str, help='Key to use in the checkpoint (example: "encoder")')
    parser.add_argument('--train_url', default=".",
                        help='physical path at OBS where to save results, also remember to copy pretrained \
                            weight under it before you launch the eval script on ModelArts.')
    parser.add_argument(
        '--data_url', default='/data/DAVIS', type=str, help='dataset root at OBS')
    parser.add_argument("--n_last_frames", type=int,
                        default=5, help="Number of preceeding frames")
    parser.add_argument("--radius", default=12, type=int,
                        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5,
                        help="accumulate label from top k neighbors")
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature softmax')
    return parser.parse_args()


def extract_feat(model: nn.Cell, frame):
    """Extract one frame feature and L2-normalize it."""
    ##Define the way your model extract feature here########################################
    feat = model(frame)  # BCHW
    ############################################################################
    l2norm_op = ops.L2Normalize(axis=1)
    feat = l2norm_op(feat)

    return feat  # BCHW


def main():
    MODELARTS_DATA_DIR = '/cache/dataset'
    MODELARTS_PRETRAINED_DIR = '/cache/pretrained'
    MODELARTS_WORK_DIR = '/cache/output'
    set_logger_level_to()
    args = add_args()
    # set to graph mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    # init dist
    rank, group_size = init_dist()
    log_info_func = partial(master_only_info, rank=rank)
    ## download dataset from obs to cache if train on ModelArts
    log_info_func('[INFO] Copying dataset from obs to ModelArts...')
    mox.file.copy_parallel(src_url=args.data_url, dst_url=MODELARTS_DATA_DIR)
    log_info_func('[INFO] Done. Start testing...')
    # dataloader
    dataset = DAVIS_VAL(MODELARTS_DATA_DIR, out_stride=args.out_stride)
    seq_names = dataset.seq_names
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, distributed=args.distribute)
    dataloader = dataloader.build_dataloader()
    ## Build your own network here #################################################
    if args.arch == 'resnet50':
        model = CustomResNet(depth=50)
        # copy pretrained weight from OBS to ModelArts
        log_info_func('[INFO] Copying saved ckpts from OBS to ModelArts...')
        mox.file.copy_parallel(src_url=args.train_url, dst_url=MODELARTS_PRETRAINED_DIR)
        log_info_func('[INFO] Done.')
        # load ckpt here
        ckpt_path = os.path.join(MODELARTS_PRETRAINED_DIR, args.pretrained_weight)
        load_partial_param_into_net(model, ckpt_path, prefix=args.encoder_key)
        # set model to test mode
        model.set_train(False)
    ################################################################################
    for seq_info, frames, first_seg, seg_ori in dataloader.create_tuple_iterator(num_epochs=1):
        # NOTE there is a batch dim when batch_size=1
        index, ori_h, ori_w = seq_info[0]
        frames = frames[0] # T*1*C*H*W
        # make save dir
        seq_name = seq_names[index]
        log_info_func(f'Processing seq {seq_name}...')
        seq_dir = os.path.join(MODELARTS_WORK_DIR, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        # extract first frame feat and saving first segmentation
        first_feat = extract_feat(model, frames[0]) 
        out_path = os.path.join(seq_dir, "00000.png")
        imwrite_indexed(out_path, seg_ori[0].asnumpy(), default_palette)
        # The queue stores the n preceeding frames
        que = []
        for frame_index in range(1, len(frames)):
            # extract current frame feat
            feat_tar = extract_feat(model, frames[frame_index])
            # we use the first segmentation and the n previous ones
            used_frame_feats = [first_feat] + [pair[0]
                                               for pair in que]
            used_segs = [first_seg] + [pair[1] for pair in que]
            q, k, v = feat_tar, msnp.stack(used_frame_feats, 2), msnp.stack(used_segs, 2)
            local_attention_mask = spatial_neighbor(q.shape[0], q.shape[-2], q.shape[-1],
                                        neighbor_range=args.radius, dtype=q.dtype)
            seg_tar = masked_attention_efficient(q, k, v, local_attention_mask,
                            args.temperature, topk=args.topk, normalize=False, step=64)
            # pop out oldest frame if neccessary
            if len(que) == args.n_last_frames:
                del que[0]
            # push current results into queue
            seg = seg_tar.copy()
            que.append([feat_tar, seg])
            # upsampling & argmax
            _h, _w = seg_tar.shape[-2:]
            resize_bilinear_op = ops.ResizeBilinear(size=(_h*args.out_stride, _w*args.out_stride))
            seg_tar = resize_bilinear_op(seg_tar)
            seg_tar = norm_mask(seg_tar[0])
            seg_tar = msnp.argmax(seg_tar, axis=0)
            # saving to disk
            seg_tar = seg_tar.asnumpy().astype(np.uint8)
            seg_tar = np.array(Image.fromarray(
                seg_tar).resize((int(ori_w.asnumpy()), int(ori_h.asnumpy())), 0))
            seg_name = os.path.join(seq_dir, f'{frame_index:05}.png')
            imwrite_indexed(seg_name, seg_tar)
    log_info_func(f'All videos has been tested, results saved at {MODELARTS_WORK_DIR}.')
    
    log_info_func('[INFO] Copying results from ModelArts to OBS...')
    mox.file.copy_parallel(src_url=MODELARTS_WORK_DIR, dst_url=args.train_url)
    log_info_func('[INFO] ALL DONE.')


if __name__ == "__main__":
    main()

