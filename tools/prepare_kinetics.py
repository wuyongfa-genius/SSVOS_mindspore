import os
from glob import glob
from tqdm import tqdm
from decord import VideoReader


def check_postfix(dataset_root):
    all_video_paths = glob(os.path.join(dataset_root, '*', '*'))
    print(f"Total video numbers:{len(all_video_paths)}")
    video_paths_ends_with_mp4 = [i for i in all_video_paths if i.endswith('mp4')]
    print(f"Mp4 video numbers:{len(video_paths_ends_with_mp4)}")

def find_long_enough_videos(dataset_root, min_video_length=8):
    """Find videos whose length is longer than min_video_length, and write their paths 
    into a txt
    
    Args:
        dataset_root(Path): root to your dataset
        min_video_length(int): the minimal frames you want to have in a video."""
    all_video_paths = glob(os.path.join(dataset_root, '*', '*.mp4'))
    enough_length_video_paths = []
    for video_path in tqdm(all_video_paths):
        try:
            with open(video_path, 'rb') as f:
                vr = VideoReader(f)
        except:
                pass
        else:
            if len(vr)>=min_video_length:
                enough_length_video_paths.append(video_path)
            del vr
    print(f"Total video numbers:{len(all_video_paths)}")
    print(f"There are {len(enough_length_video_paths)} intact videos in total.")
    enough_length_video_paths_txt = os.path.join(dataset_root, f'length_over_{min_video_length}_video_paths.txt')
    with open(enough_length_video_paths_txt, 'w') as f:
        f.write('\n'.join(enough_length_video_paths))
    print(f"Have written length_over_{min_video_length} video paths into file {enough_length_video_paths_txt}.")


if __name__=="__main__":
    # check_postfix('/data/datasets/Kinetics-400/train_256')
    find_long_enough_videos('/data/datasets/Kinetics-400/train_256')