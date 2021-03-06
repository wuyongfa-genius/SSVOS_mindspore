"""Evaluation script on VOS datasets. Support multi-gpu test."""
import argparse
import os

import numpy as np
from PIL import Image
from mindspore import nn, numpy as msnp, ops, context, log as logger
from ssvos.datasets import (DAVIS_VAL, imwrite_indexed, norm_mask, default_palette,
                            DataLoader)
from ssvos.models.backbones import ResNet, CustomResNet
from ssvos.utils.dist_utils import init_dist
from ssvos.utils.attention import spatial_neighbor, masked_attention_efficient


def add_args():
    parser = argparse.ArgumentParser(
        'Evaluation with video object segmentation on DAVIS 2017')
    parser.add_argument('--device_target', type=str, default='GPU',
                        help='Device target, Currently GPU, Ascend are supported.')
    parser.add_argument('--distribute', type=bool, default=True,
                        help='Run distributed testing.')
    parser.add_argument('--pretrained_weights', default='',
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['deit_tiny', 'deit_small', 'resnet18', 'resnet50'], help='Architectures.')
    parser.add_argument('--out_stride', default=8, type=int,
                        help='Output stride of the model.')
    parser.add_argument("--encoder_key", default="online_encoder.encoder",
                        type=str, help='Key to use in the checkpoint (example: "encoder")')
    parser.add_argument('--output_dir', default=".",
                        help='Path where to save segmentations')
    parser.add_argument(
        '--data_path', default='/data/DATASETS/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS', type=str)
    parser.add_argument("--n_last_frames", type=int,
                        default=5, help="Number of preceeding frames")
    parser.add_argument("--radius", default=12, type=int,
                        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5,
                        help="accumulate label from top k neighbors")
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature softmax')
    # parser.add_argument("--propagation_type", default='soft', choices=['soft', 'hard'],
    #                     help="Whether to quantize the predicted seg. `hard` means quantize.")
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
    args = add_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    if args.device_target != "Ascend" and args.device_target != "GPU":
        raise ValueError("Unsupported device target.")
    if args.distribute:
        # init dist
        rank, group_size = init_dist()
    else:
        rank, group_size = 0, 1
    # dataloader
    dataset = DAVIS_VAL(args.data_path, out_stride=args.out_stride)
    seq_names = dataset.seq_names
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, distributed=args.distribute)
    dataloader = dataloader.build_dataloader()
    ## Build your own network here #################################################
    if args.arch == 'resnet50':
        model = CustomResNet(depth=50)
        # load ckpt here
        # load weights into model here
        # set model to test mode
        model.set_train(False)
        # NOTE not sure that GRAPH_MODE needs this
        model.set_grad(False)
    ################################################################################
    logger.info('Start testing...')
    for seq_info, frames, first_seg, seg_ori in dataloader.create_tuple_iterator(num_epochs=1):
        # NOTE there is a batch dim when batch_size=1
        index, ori_h, ori_w = seq_info[0]
        frames = frames[0] # T*1*C*H*W
        # make save dir
        seq_name = seq_names[index]
        seq_dir = os.path.join(args.output_dir, seq_name)
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
                            args.temperature, topk=args.topk, step=64)
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
    logger.info(f'All videos has been tested, results saved at {args.output_dir}.')


if __name__ == "__main__":
    main()

