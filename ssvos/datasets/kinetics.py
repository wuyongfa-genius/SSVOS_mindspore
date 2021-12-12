"""Dataset to load kinetics-400 dataset."""
import os

import numpy as np
from PIL import Image
from decord import VideoReader
from .transforms import (Compose, RandomResizedCrop, RandomHorizontalFlip,
                         ToTensor, Normalize)


class Kinetics:
    """The dataset class to load Kinetics-400 Dataset.

    Args:
        root(Path): root path of the dataset
        ann_file(Path): path to the annotation file in which is the
            video paths
        num_frames(int): how many frames to sample in a clip."""

    def __init__(self,
                 root,
                 ann_file,
                 num_frames=8,
                 frame_interval=8,
                 out_of_bound_opt='loop'):
        self.root = root
        self.ann_file = ann_file
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.out_of_bound_opt = out_of_bound_opt

        self.all_video_paths = self._get_all_video_paths()

        self.transforms = Compose([
            RandomResizedCrop(size=224, scale=(0.2, 1)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize()
        ])

    def _get_all_video_paths(self):
        all_video_paths = []
        with open(os.path.join(self.root, self.ann_file), 'r') as f:
            lines = f.readlines()
        all_video_paths = [line.strip() for line in lines]

        return all_video_paths

    def _init_video_reader(self, file_path, num_threads=4):
        with open(file_path, 'rb') as f:
            container = VideoReader(f, num_threads=num_threads)

        return container

    def _get_clip_offsets(self, num_frames, num_clips=2):
        """Copied from mmaction2. Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.num_frames * self.frame_interval

        avg_interval = (num_frames - ori_clip_len + 1) // num_clips

        if avg_interval > 0:
            base_offsets = np.arange(num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=num_clips)
        elif num_frames > max(num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / num_clips
            clip_offsets = np.around(np.arange(num_clips) * ratio)
        else:
            clip_offsets = np.zeros((num_clips, ), dtype=np.int)

        return clip_offsets

    def __getitem__(self, idx):
        video_path = self.all_video_paths[idx]
        video_full_path = os.path.join(self.root, video_path)
        container = self._init_video_reader(video_full_path)
        total_frames = len(container)
        # get clip offsets
        clip_offsets = self._get_clip_offsets(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.num_frames)[None, :] * self.frame_interval
        # get frame indices
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        else:
            raise ValueError('Illegal out_of_bound option.')
        # load frame
        frame_inds = np.squeeze(frame_inds)
        frames = container.get_batch(frame_inds).asnumpy()  # N,H,W,3
        del container # remember to delete container
        _clip1, _clip2 = frames[:self.num_frames], frames[self.num_frames:]
        # augmentation
        clip1 = [Image.fromarray(frame) for frame in _clip1]
        clip2 = [Image.fromarray(frame) for frame in _clip2]
        transformed_clip1 = self.transforms(clip1)
        transformed_clip2 = self.transforms(clip2)

        # CTHW
        return np.stack(transformed_clip1, 1), np.stack(transformed_clip2, 1)

    def __len__(self):
        return len(self.all_video_paths)


