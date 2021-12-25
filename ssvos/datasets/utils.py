"""Some dataset or transform utils adapted from MindSpore."""
from PIL import Image

import os
import numpy as np
from mindspore.dataset.vision import Inter
from mindspore.dataset import GeneratorDataset
from mindspore import communication as dist

augment_error_message = "img should be PIL image. Got {}. Use Decode() for encoded data or ToPIL() for decoded data."

PATH_PALETTE = 'ssvos/datasets/palette.txt'
default_palette = np.loadtxt(PATH_PALETTE, dtype=np.uint8).reshape(-1, 3)

def is_pil(img):
    """
    Check if the input image is PIL format.

    Args:
        img: Image to be checked.

    Returns:
        Bool, True if input is PIL image.
    """
    return isinstance(img, Image.Image)


def crop(img, top, left, height, width):
    """
    Crop the input PIL image.

    Args:
        img (PIL image): Image to be cropped. (0,0) denotes the top left corner of the image,
            in the directions of (width, height).
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        img (PIL image), Cropped image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))

    return img.crop((left, top, left + width, top + height))


def resize(img, size, interpolation=Inter.BILINEAR):
    """
    Resize the input PIL image to desired size.

    Args:
        img (PIL image): Image to be resized.
        size (Union[int, sequence]): The output size of the resized image.
            If size is an integer, smaller edge of the image will be resized to this value with
            the same image aspect ratio.
            If size is a sequence of (height, width), this will be the desired output size.
        interpolation (interpolation mode): Image interpolation mode. Default is Inter.BILINEAR = 2.

    Returns:
        img (PIL image), Resized image.
    """
    if not is_pil(img):
        raise TypeError(augment_error_message.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) == 2)):
        raise TypeError('Size should be a single number or a list/tuple (h, w) of length 2.'
                        'Got {}.'.format(size))

    if isinstance(size, int):
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height  # maintain the aspect ratio
        if (img_width <= img_height and img_width == size) or \
                (img_height <= img_width and img_height == size):
            return img
        if img_width < img_height:
            out_width = size
            out_height = int(size / aspect_ratio)
            return img.resize((out_width, out_height), interpolation)
        out_height = size
        out_width = int(size * aspect_ratio)
        return img.resize((out_width, out_height), interpolation)
    return img.resize(size[::-1], interpolation)


class DataLoader:
    def __init__(self,
                dataset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                drop_last=False,
                transforms=None,
                column_names=[],
                distributed=False,
                **kwargs
                ):      
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transforms = transforms
        self.column_names = column_names
        self.distributed = distributed

        self.max_rowsize = kwargs.get('max_rowsize', 16)
    
    def build_dataloader(self):
        # get or set col names
        col_names = self.column_names
        if len(col_names)==0:
            example_item = self.dataset[0]
            item_len = len(example_item)
            col_names = [f'col_{i}' for i in range(item_len)]
        if self.distributed:
            rank_size = dist.get_group_size()
            rank_id = dist.get_rank()
            data_generator = GeneratorDataset(self.dataset,
                                            column_names=col_names,
                                            num_parallel_workers=self.num_workers,
                                            shuffle=self.shuffle,
                                            num_shards=rank_size,
                                            shard_id=rank_id,
                                            max_rowsize=self.max_rowsize)
        else:
            data_generator = GeneratorDataset(self.dataset,
                                            column_names=col_names,
                                            num_parallel_workers=self.num_workers,
                                            shuffle=self.shuffle,
                                            max_rowsize=self.max_rowsize)
        if self.transforms is not None:
            data_generator = data_generator.map(operations=self.transforms,
                                                num_parallel_workers=self.num_workers,
                                                max_rowsize=self.max_rowsize)
        dataloader = data_generator.batch(batch_size=self.batch_size,
                                    drop_remainder=self.drop_last,
                                    max_rowsize=self.max_rowsize)
        
        return dataloader


def imwrite_indexed(filename, array, color_palette=default_palette):
    """ Save indexed png."""

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def norm_mask(mask):
    c, h, w = mask.shape
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


# if __name__=="__main__":
#     from ssvos.utils.dist_utils import init_dist
#     init_dist()

    # from ssvos.datasets.video_dataset import RawFrameDataset
    # ytvos_dataset = RawFrameDataset(root='/data/DATASETS/Youtube_VOS/2018', ann_file='ytvos_2018_raw_frames.txt')
    # dataloader = DataLoader(ytvos_dataset, 2, 4, shuffle=True)
    # dataloader = dataloader.build_dataloader()

    # for batch in dataloader.create_tuple_iterator():
    #     clip1, clip2 = batch
    #     print(f'{type(clip1)}, {clip1.shape}')
    #     print(f'{type(clip2)}, {clip2.shape}')

    # from ssvos.datasets.davis import DAVIS_VAL
    # davis_dataset = DAVIS_VAL('/data/DATASETS/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS')
    # dataloader = DataLoader(davis_dataset, 1, 1, column_names=['seq_info', 'frames', 'small_seg', 'seg_ori'], distributed=True)
    # dataloader = dataloader.build_dataloader()

    # batch_iter = dataloader.create_tuple_iterator()
    # for _ in range(30):
    #     batch = next(batch_iter)
    #     seq_info, frames, small_seg, seg_ori = batch
    #     print(f'index: {seq_info[0][0]}')
    #     print(f'frame.shape: {frames[0].shape}')
    #     print(f'h: {seq_info[0][1]}, w: {seq_info[0][2]}')
    #     print(f'small_seg.shape: {small_seg.shape}')
    #     print(f'seg_ori.shape: {seg_ori.shape}')
