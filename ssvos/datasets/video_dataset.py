"""Dataset to load kinetics-400 dataset."""
# from decord import VideoReader
import os

import numpy as np
from PIL import Image
from ssvos.datasets.opencv_video_reader import VideoReader
# from decord import VideoReader
from ssvos.datasets.transforms import (Compose, RandomResizedCrop, RandomHorizontalFlip,
                         ToTensor, Normalize)


def _get_clip_offsets(num_frames, clip_len, frame_interval=1, num_clips=1):
    """Copied from mmaction2. Get clip offsets in train mode.

    It will calculate the average interval for selected frames,
    and randomly shift them within offsets between [0, avg_interval].
    If the total number of frames is smaller than clips num or origin
    frames length, it will return all zero indices.

    Args:
        num_frames (int): Total number of frames in the video.
        clip_len(int): number of frames in one sampled clip.
        frame_interval(int): interval between two adjacent sampled frames.
        num_clips(int): number of clips to be sampled.

    Returns:
        np.ndarray: Sampled frame indices in train mode.
    """
    ori_clip_len = clip_len * frame_interval

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

    return clip_offsets.astype(np.int64)


class VideoDataset:
    """The dataset class to load Datasets containing video files.

    Args:
        root(Path): root path of the dataset
        ann_file(Path): path to the annotation file in which is the
            video paths
        num_frames(int): how many frames to sample in a clip.
        frame_interval(int): interval between two adjacent sampled frames.
        out_of_bound_opt(str): the way to deal with out of bound sampled 
            frame indices. `loop` means choose from beginning, `repeat` means
            repeat the last frame index."""

    def __init__(self,
                 root,
                 ann_file,
                 num_frames=8,
                 frame_interval=8,
                 out_of_bound_opt='loop',
                 video_reader='opencv'):
        assert video_reader in ['opencv', 'decord']
        self.root = root
        self.ann_file = ann_file
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.out_of_bound_opt = out_of_bound_opt
        self.video_reader = video_reader

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

    def _init_decord_video_reader(self, file_path, num_threads=4):
        with open(file_path, 'rb') as f:
            container = VideoReader(f, num_threads=num_threads)

        return container
    
    def _init_opencv_video_reader(self, file_path, cache_capacity=10):
        return VideoReader(file_path, cache_capacity=cache_capacity)

    def __getitem__(self, idx):
        video_path = self.all_video_paths[idx]
        video_full_path = os.path.join(self.root, video_path)
        if self.video_reader == 'opencv':
            container = self._init_opencv_video_reader(video_full_path)
        elif self.video_reader == 'decord':
            container = self._init_decord_video_reader(video_full_path)
        total_frames = len(container)
        # get clip offsets
        clip_offsets = _get_clip_offsets(total_frames, self.num_frames, self.frame_interval, 2)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.num_frames)[None, :] * self.frame_interval
        # get frame indices
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        else:
            raise ValueError('Illegal out_of_bound option.')
        # load frame
        frame_inds = np.squeeze(frame_inds)
        if self.video_reader == 'opencv':
            ### opencv video reader ##################################
            frames = []
            for frame_ind in frame_inds:
                cur_frame = container[frame_ind]
                # last frame may be None in OpenCV
                while isinstance(cur_frame, type(None)):
                    frame_ind -= 1
                    cur_frame = container[frame_ind]
                frames.append(cur_frame)
            frames = np.array(frames)
            frames = frames[:, :, :, ::-1] # BGR2RGB
        elif self.video_reader == 'decord':
            ### decord video reader ###################################
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


class RawFrameDataset:
    """The dataset class to load Datasets containing raw frame images.

    Args:
        root(Path): root path of the dataset.
        ann_file(Path): path to the annotation file in which is the
            directories containing raw frame images.
        num_frames(int): how many frames to sample in a clip.
        frame_interval(int): interval between two adjacent sampled frames.
        out_of_bound_opt(str): the way to deal with out of bound sampled 
            frame indices. `loop` means choose from beginning, `repeat` means
            repeat the last frame index."""

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

        self.all_frame_dirs = self._get_all_frame_dirs()

        self.transforms = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.6, 1)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize()
        ])
    
    def _get_all_frame_dirs(self):
        all_frame_dirs = []
        with open(os.path.join(self.root, self.ann_file), 'r') as f:
            lines = f.readlines()
        all_frame_dirs = [line.strip() for line in lines]

        return all_frame_dirs

    def _load_frames(self, frame_dir, frame_names, frame_inds):
        sampled_frame_names = [frame_names[i] for i in frame_inds]
        sampled_frames = []
        for frame_name in sampled_frame_names:
            frame_path = os.path.join(frame_dir, frame_name)
            sampled_frames.append(Image.open(frame_path))
        
        return sampled_frames

    def __getitem__(self, idx):
        frame_dir_wrt_datasetroot = self.all_frame_dirs[idx]
        frame_dir_full_path = os.path.join(self.root, frame_dir_wrt_datasetroot)
        frame_names = sorted(os.listdir(frame_dir_full_path))
        total_frames = len(frame_names)
        # get clip offsets
        clip_offsets = _get_clip_offsets(total_frames, self.num_frames, self.frame_interval, 2)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.num_frames)[None, :] * self.frame_interval
        # get frame indices
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        else:
            raise ValueError('Illegal out_of_bound option.')
        # load frame
        frame_inds = np.ravel(frame_inds)
        frames = self._load_frames(frame_dir_full_path, frame_names, frame_inds)
        clip1, clip2 = frames[:self.num_frames], frames[self.num_frames:]
        # augmentation
        transformed_clip1 = self.transforms(clip1)
        transformed_clip2 = self.transforms(clip2)

        # CTHW
        return np.stack(transformed_clip1, 1), np.stack(transformed_clip2, 1)

    def __len__(self):
        return len(self.all_frame_dirs)

# if __name__=="__main__":
#     import random
#     ytvos_dataset = RawFrameDataset(root='/data/DATASETS/Youtube_VOS/2018', ann_file='ytvos_2018_raw_frames.txt')
#     length = len(ytvos_dataset)
#     chosen_idx = random.sample(range(length), k=10)
#     for idx in chosen_idx:
#         clip1, clip2 = ytvos_dataset[idx]
#         print(clip1.shape)
#         print(clip2.shape)
