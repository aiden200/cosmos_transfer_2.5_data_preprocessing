from pipeline.control_net_generation.segment_videos import run_sam2_pipeline
import os
import cv2
import numpy as np
import tempfile
import subprocess
import shlex


def run_mask_pipeline(input_file, box, output_video):
    # 1. Temporary file to store raw segmentation output from SAM2
    # temp_video_location = "pipeline/test.mp4"
    temp_dir = tempfile.gettempdir()
    temp_video_location = os.path.join(temp_dir, f"{os.path.basename(input_file[:-4])}_temp_mask.mp4")

    # 2. Run SAM2 segmentation (this generates a video where the detected objects are non-black and the rest is black)

    run_sam2_pipeline(input_file, box, temp_video_location, box=True)    
    
    ffmpeg_cmd = f'''
    ffmpeg -y -i "{temp_video_location}" \
        -vf "format=gray,lut='if(gt(val\,0)\,255\,0)'" \
        -c:v libx264 -pix_fmt yuv420p "{output_video}"
    '''
    process = subprocess.run(
        shlex.split(ffmpeg_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if process.returncode != 0:
        print("Error:", process.stderr.decode())
   
    
   
    ffmpeg_cmd = f'''
    ffmpeg -y -i {output_video} \
    -vf "negate" \
    -c:v libx264 -pix_fmt yuv420p {output_video[:-4]}_inversion.mp4
    '''
    process = subprocess.run(shlex.split(ffmpeg_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        print("Error:", process.stderr.decode())

    if os.path.exists(temp_video_location):
        os.remove(temp_video_location)

    # print(f"[MASK] Saved final binary mask to {output_video}")