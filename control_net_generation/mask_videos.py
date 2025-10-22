from pipeline.control_net_generation.segment_videos import run_sam2_pipeline
import os
import cv2
import numpy as np
import tempfile



def run_mask_pipeline(input_file, prompt_items, output_video):
    # 1. Temporary file to store raw segmentation output from SAM2
    temp_dir = tempfile.gettempdir()
    temp_video_location = os.path.join(temp_dir, f"{os.path.basename(input_file)}_temp_mask.mp4")

    # 2. Run SAM2 segmentation (this generates a video where the detected objects are non-black and the rest is black)
    run_sam2_pipeline(input_file, prompt_items, temp_video_location)

    # 3. Open the temp video and read frames
    cap = cv2.VideoCapture(temp_video_location)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open temporary SAM2 video at {temp_video_location}")

    # Get video resolution and FPS to write output in the same format
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 4. Set up VideoWriter for final mask output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)

    # 5. Convert non-black pixels → white mask (255), black stays 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # If frame is color, convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Anything not black → 255 (white)
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Write to output
        out.write(binary_mask)

    cap.release()
    out.release()

    # 6. (Optional) Remove temporary file
    if os.path.exists(temp_video_location):
        os.remove(temp_video_location)

    # print(f"[MASK] Saved final binary mask to {output_video}")