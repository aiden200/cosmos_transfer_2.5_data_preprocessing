from pipeline.control_net_generation.segment_videos import run_sam2_pipeline
import os
import cv2
import numpy as np
import tempfile
import subprocess
import shlex


def run_mask_pipeline(input_file, prompt_items, output_video):
    # 1. Temporary file to store raw segmentation output from SAM2
    # temp_video_location = "pipeline/test.mp4"
    temp_dir = tempfile.gettempdir()
    temp_video_location = os.path.join(temp_dir, f"{os.path.basename(input_file)}_temp_mask.mp4")
    temp_tensor_location = os.path.join(temp_dir, f"{os.path.basename(input_file)}_temp_tensor.pt")

    # 2. Run SAM2 segmentation (this generates a video where the detected objects are non-black and the rest is black)

    run_sam2_pipeline(input_file, prompt_items, temp_video_location, temp_tensor_location)

    # 3. Open the temp video and read frames
    cap = cv2.VideoCapture(temp_video_location)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open temporary SAM2 video at {temp_video_location}")
    

    ffmpeg_cmd = f'''
    ffmpeg -y -i "{input_file}" \
        -vf "format=gray,scale=trunc(iw/2)*2:trunc(ih/2)*2,lut=if(gt(val\\,1)\\,255\\,0)" \
        -c:v libx264 -pix_fmt yuv420p "{output_video[:-4]}_inversion.mp4"
    '''
    process = subprocess.run(shlex.split(ffmpeg_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        print("Error:", process.stderr.decode())
    
    
    ffmpeg_cmd = f'''
    ffmpeg -y -i "{input_file}" \
        -vf "format=gray,threshold=1,lutyuv=y=negval" \
        -c:v libx264 -pix_fmt yuv420p "{output_video[:-4]}"
    '''
    process = subprocess.run(shlex.split(ffmpeg_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        print("Error:", process.stderr.decode())

    if os.path.exists(temp_video_location):
        os.remove(temp_video_location)

    # print(f"[MASK] Saved final binary mask to {output_video}")