"""Implement some video clip transformation."""
import random
import math
from PIL import Image
import numpy as np
from .utils import crop, resize
from mindspore.dataset.vision import py_transforms as VPT, Inter
from mindspore.dataset.transforms import py_transforms as TPT


class Compose(TPT.PyTensorOperation):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
    
    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)

        return clip


class RandomResizedCrop(TPT.PyTensorOperation):
    """Randomly crop every frame in the clip and resize it to a given size.

    Args:
        size (Union[int, sequence]): The size of the output image.
            If size is an integer, a square of size (size, size) is returned.
            If size is a sequence of length 2, it should be in shape of (height, width).
        scale (Union[list, tuple], optional): Respective size range of the original image to be cropped
            in shape of (min, max) (default=(0.08, 1.0)).
        ratio (Union[list, tuple], optional): Aspect ratio range to be cropped
            in shape of (min, max) (default=(3./4., 4./3.)).
        interpolation (Inter, optional): Image interpolation mode (default=Inter.BILINEAR).
        max_attempts (int, optional): The maximum number of attempts to propose a valid
            crop area (default=10). If exceeded, fall back to use center crop instead."""

    def __init__(self, size=224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Inter.BILINEAR, same_on_frames=True):
        assert 0 < scale[0] <= scale[1] <= 1
        assert 0 < ratio[0] <= ratio[1]
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.same_on_frames = same_on_frames

    @staticmethod
    def _get_crop_param(img_shape, scale, ratio, max_attempts=10):
        img_width, img_height = img_shape
        img_area = img_width * img_height

        for _ in range(max_attempts):
            crop_area = random.uniform(scale[0], scale[1]) * img_area
            # in case of non-symmetrical aspect ratios,
            # use uniform distribution on a logarithmic scale.
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            width = int(round(math.sqrt(crop_area * aspect_ratio)))
            height = int(round(width / aspect_ratio))

            if 0 < width <= img_width and 0 < height <= img_height:
                top = random.randint(0, img_height - height)
                left = random.randint(0, img_width - width)
                return top, left, height, width

        # exceeding max_attempts, use center crop
        img_ratio = img_width / img_height
        if img_ratio < ratio[0]:
            width = img_width
            height = int(round(width / ratio[0]))
        elif img_ratio > ratio[1]:
            height = img_height
            width = int(round(height * ratio[1]))
        else:
            width = img_width
            height = img_height
        top = int(round((img_height - height) / 2.))
        left = int(round((img_width - width) / 2.))
        return top, left, height, width

    def __call__(self, clip):
        assert isinstance(clip, (list, tuple)) and isinstance(clip[0], Image.Image)
        image_shape = clip[0].size # w, h
        cropped_frames = []
        if self.same_on_frames:
            top, left, height, width = self._get_crop_param(image_shape, self.scale, self.ratio)
            for frame in clip:
                frame = crop(frame, top, left, height, width)
                frame = resize(frame, self.size, self.interpolation)
                cropped_frames.append(frame)
        else:
            for frame in clip:
                top, left, height, width = self._get_crop_param(image_shape, self.scale, self.ratio)
                frame = crop(frame, top, left, height, width)
                frame = resize(frame, self.size, self.interpolation)
                cropped_frames.append(frame)

        return cropped_frames


class RandomHorizontalFlip(TPT.PyTensorOperation):
    """Randomly flip the frames in the clip horizontally with a given probability.

    Args:
        prob (float, optional): Probability of the image to be horizontally flipped (default=0.5)."""
    def __init__(self, prob=0.5, same_on_frames=True):
        super().__init__()
        assert prob>=0. and prob<=1.
        self.prob = prob
        self.same_on_frames = same_on_frames
    
    def __call__(self, clip):
        assert isinstance(clip, (list, tuple)) and isinstance(clip[0], Image.Image)
        flipped_frames = []
        if self.same_on_frames:
            if self.prob > random.random():
                for frame in clip:
                    flipped_frames.append(frame.transpose(Image.FLIP_LEFT_RIGHT))
            else:
                flipped_frames = clip
        else:
            for frame in clip:
                if self.prob > random.random():
                    flipped_frames.append(frame.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    flipped_frames.append(frame)

        return flipped_frames


class ToTensor(VPT.ToTensor):
    """Convert the input PIL Image clip or sequence of numpy.ndarray of shape (H, W, C) in the range [0, 255] to numpy.ndarray of
    shape (C, H, W) in the range [0.0, 1.0] with the desired dtype.
    
    Args:
        output_type (numpy.dtype, optional): The dtype of the numpy.ndarray output (default=np.float32)."""
    def __init__(self, output_type=np.float32):
        super().__init__(output_type=output_type)
    
    def __call__(self, clip):
        assert isinstance(clip, (list, tuple)) and isinstance(clip[0], Image.Image)
        tensor_clip = []
        for frame in clip:
            tensor_clip.append(super().__call__(frame))
        
        return tensor_clip


class Normalize(VPT.Normalize):
    """Normalize the input numpy.ndarray image of shape (C, H, W) with the specified mean and standard deviation.
    Args:
        mean (Union[float, sequence]): list or tuple of mean values for each channel, arranged in channel order. The
            values must be in the range [0.0, 1.0].
            If a single float is provided, it will be filled to the same length as the channel.
        std (Union[float, sequence]): list or tuple of standard deviation values for each channel, arranged in channel
            order. The values must be in the range (0.0, 1.0].
            If a single float is provided, it will be filled to the same length as the channel."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__(mean=mean, std=std)
    
    def __call__(self, clip):
        tensor_clip = []
        for frame in clip:
            tensor_clip.append(super().__call__(frame))
        
        return tensor_clip
