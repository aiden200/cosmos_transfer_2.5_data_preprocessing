import torchvision.transforms as TS
import cv2, os


def get_filename_no_suffix(video_path):
    base = os.path.basename(video_path)    
    name, _ = os.path.splitext(base)            
    return name


def create_folders(video_filenames):
    os.makedirs("pipeline/outputs", exist_ok=True)
    for video_filename in video_filenames:
        parent_dir = os.path.join("pipeline/outputs", get_filename_no_suffix(video_filename))
        os.makedirs(parent_dir, exist_ok=True)


def find_video_tags(video_names, control_net_type):
    valid_video_names = []
    for filename in video_names:
        name = get_filename_no_suffix(filename)
        parent_dir = os.path.join("pipeline/outputs", name)
        if not os.path.exists(os.path.join(parent_dir, control_net_type)):
            valid_video_names.append(filename)
    
    return valid_video_names