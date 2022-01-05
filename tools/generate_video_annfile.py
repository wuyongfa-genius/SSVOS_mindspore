import os


def generate_ytvos_annfile(root, version='2018'):
    raw_frame_parent_dir = os.path.join(root, version, 'train_all_frames', 'JPEGImages')
    print(f'[INFO] Search {raw_frame_parent_dir} for raw frames...')
    all_frame_dirs = os.listdir(raw_frame_parent_dir)
    annfile_dir = os.path.join(root, version)
    annfile_path = os.path.join(annfile_dir, f'ytvos_{version}_raw_frames.txt')
    print(f'[INFO] Start writing frame dirs to {annfile_path}...')
    with open(annfile_path, 'w') as f:
        for frame_dir in all_frame_dirs:
            frame_dir_wrt_root = os.path.join('train_all_frames', 'JPEGImages', frame_dir)
            f.write(f'{frame_dir_wrt_root}\n')
    print('[INFO] Done.')

def generate_davis_annfile(root):
    raw_frame_parent_dir = os.path.join(root, 'JPEGImages', '480p')
    print(f'[INFO] Search {raw_frame_parent_dir} for raw frames...')
    all_frame_dirs = os.listdir(raw_frame_parent_dir)
    annfile_dir = root
    annfile_path = os.path.join(annfile_dir, f'davis_raw_frames.txt')
    print(f'[INFO] Start writing frame dirs to {annfile_path}...')
    with open(annfile_path, 'w') as f:
        for frame_dir in all_frame_dirs:
            frame_dir_wrt_root = os.path.join('JPEGImages', '480p', frame_dir)
            f.write(f'{frame_dir_wrt_root}\n')
    print('[INFO] Done.')
    

if __name__=="__main__":
    # generate_ytvos_annfile(root='/data/DATASETS/Youtube_VOS')
    generate_davis_annfile('/data/DATASETS/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS')