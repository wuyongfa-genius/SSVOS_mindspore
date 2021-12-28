"""Some dataset or transform utils adapted from MindSpore."""
from PIL import Image

import os
import numpy as np
from mindspore.dataset.vision import Inter
from mindspore.dataset import GeneratorDataset
from mindspore import communication as dist

augment_error_message = "img should be PIL image. Got {}. Use Decode() for encoded data or ToPIL() for decoded data."

default_palette = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [191, 0, 0],
        [64, 128, 0],
        [191, 128, 0],
        [64, 0, 128],
        [191, 0, 128],
        [64, 128, 128],
        [191, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 191, 0],
        [128, 191, 0],
        [0, 64, 128],
        [128, 64, 128],
        [22, 22, 22],
        [23, 23, 23],
        [24, 24, 24],
        [25, 25, 25],
        [26, 26, 26],
        [27, 27, 27],
        [28, 28, 28],
        [29, 29, 29],
        [30, 30, 30],
        [31, 31, 31],
        [32, 32, 32],
        [33, 33, 33],
        [34, 34, 34],
        [35, 35, 35],
        [36, 36, 36],
        [37, 37, 37],
        [38, 38, 38],
        [39, 39, 39],
        [40, 40, 40],
        [41, 41, 41],
        [42, 42, 42],
        [43, 43, 43],
        [44, 44, 44],
        [45, 45, 45],
        [46, 46, 46],
        [47, 47, 47],
        [48, 48, 48],
        [49, 49, 49],
        [50, 50, 50],
        [51, 51, 51],
        [52, 52, 52],
        [53, 53, 53],
        [54, 54, 54],
        [55, 55, 55],
        [56, 56, 56],
        [57, 57, 57],
        [58, 58, 58],
        [59, 59, 59],
        [60, 60, 60],
        [61, 61, 61],
        [62, 62, 62],
        [63, 63, 63],
        [64, 64, 64],
        [65, 65, 65],
        [66, 66, 66],
        [67, 67, 67],
        [68, 68, 68],
        [69, 69, 69],
        [70, 70, 70],
        [71, 71, 71],
        [72, 72, 72],
        [73, 73, 73],
        [74, 74, 74],
        [75, 75, 75],
        [76, 76, 76],
        [77, 77, 77],
        [78, 78, 78],
        [79, 79, 79],
        [80, 80, 80],
        [81, 81, 81],
        [82, 82, 82],
        [83, 83, 83],
        [84, 84, 84],
        [85, 85, 85],
        [86, 86, 86],
        [87, 87, 87],
        [88, 88, 88],
        [89, 89, 89],
        [90, 90, 90],
        [91, 91, 91],
        [92, 92, 92],
        [93, 93, 93],
        [94, 94, 94],
        [95, 95, 95],
        [96, 96, 96],
        [97, 97, 97],
        [98, 98, 98],
        [99, 99, 99],
        [100, 100, 100],
        [101, 101, 101],
        [102, 102, 102],
        [103, 103, 103],
        [104, 104, 104],
        [105, 105, 105],
        [106, 106, 106],
        [107, 107, 107],
        [108, 108, 108],
        [109, 109, 109],
        [110, 110, 110],
        [111, 111, 111],
        [112, 112, 112],
        [113, 113, 113],
        [114, 114, 114],
        [115, 115, 115],
        [116, 116, 116],
        [117, 117, 117],
        [118, 118, 118],
        [119, 119, 119],
        [120, 120, 120],
        [121, 121, 121],
        [122, 122, 122],
        [123, 123, 123],
        [124, 124, 124],
        [125, 125, 125],
        [126, 126, 126],
        [127, 127, 127],
        [128, 128, 128],
        [129, 129, 129],
        [130, 130, 130],
        [131, 131, 131],
        [132, 132, 132],
        [133, 133, 133],
        [134, 134, 134],
        [135, 135, 135],
        [136, 136, 136],
        [137, 137, 137],
        [138, 138, 138],
        [139, 139, 139],
        [140, 140, 140],
        [141, 141, 141],
        [142, 142, 142],
        [143, 143, 143],
        [144, 144, 144],
        [145, 145, 145],
        [146, 146, 146],
        [147, 147, 147],
        [148, 148, 148],
        [149, 149, 149],
        [150, 150, 150],
        [151, 151, 151],
        [152, 152, 152],
        [153, 153, 153],
        [154, 154, 154],
        [155, 155, 155],
        [156, 156, 156],
        [157, 157, 157],
        [158, 158, 158],
        [159, 159, 159],
        [160, 160, 160],
        [161, 161, 161],
        [162, 162, 162],
        [163, 163, 163],
        [164, 164, 164],
        [165, 165, 165],
        [166, 166, 166],
        [167, 167, 167],
        [168, 168, 168],
        [169, 169, 169],
        [170, 170, 170],
        [171, 171, 171],
        [172, 172, 172],
        [173, 173, 173],
        [174, 174, 174],
        [175, 175, 175],
        [176, 176, 176],
        [177, 177, 177],
        [178, 178, 178],
        [179, 179, 179],
        [180, 180, 180],
        [181, 181, 181],
        [182, 182, 182],
        [183, 183, 183],
        [184, 184, 184],
        [185, 185, 185],
        [186, 186, 186],
        [187, 187, 187],
        [188, 188, 188],
        [189, 189, 189],
        [190, 190, 190],
        [191, 191, 191],
        [192, 192, 192],
        [193, 193, 193],
        [194, 194, 194],
        [195, 195, 195],
        [196, 196, 196],
        [197, 197, 197],
        [198, 198, 198],
        [199, 199, 199],
        [200, 200, 200],
        [201, 201, 201],
        [202, 202, 202],
        [203, 203, 203],
        [204, 204, 204],
        [205, 205, 205],
        [206, 206, 206],
        [207, 207, 207],
        [208, 208, 208],
        [209, 209, 209],
        [210, 210, 210],
        [211, 211, 211],
        [212, 212, 212],
        [213, 213, 213],
        [214, 214, 214],
        [215, 215, 215],
        [216, 216, 216],
        [217, 217, 217],
        [218, 218, 218],
        [219, 219, 219],
        [220, 220, 220],
        [221, 221, 221],
        [222, 222, 222],
        [223, 223, 223],
        [224, 224, 224],
        [225, 225, 225],
        [226, 226, 226],
        [227, 227, 227],
        [228, 228, 228],
        [229, 229, 229],
        [230, 230, 230],
        [231, 231, 231],
        [232, 232, 232],
        [233, 233, 233],
        [234, 234, 234],
        [235, 235, 235],
        [236, 236, 236],
        [237, 237, 237],
        [238, 238, 238],
        [239, 239, 239],
        [240, 240, 240],
        [241, 241, 241],
        [242, 242, 242],
        [243, 243, 243],
        [244, 244, 244],
        [245, 245, 245],
        [246, 246, 246],
        [247, 247, 247],
        [248, 248, 248],
        [249, 249, 249],
        [250, 250, 250],
        [251, 251, 251],
        [252, 252, 252],
        [253, 253, 253],
        [254, 254, 254],
        [255, 255, 255],
    ]
)


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
        if len(col_names) == 0:
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
                                                num_parallel_workers=self.num_workers)
        dataloader = data_generator.batch(batch_size=self.batch_size,
                                          drop_remainder=self.drop_last)

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
