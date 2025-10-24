import os

import numpy as np

import torch
import cv2
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image, ImageDraw
import numpy as np
import tempfile
import subprocess
import shlex



def compute_box_centers(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    return_str = ""
    labels = ""
    for box in boxes:
        x_centers = (boxes[0] + boxes[2]) // 2
        y_centers = (boxes[1] + boxes[3]) // 2
        return_str += f"{x_centers[0]},{y_centers[0]};"
        labels += "1,"
    return return_str[:-1], labels[:-1]


def convert_normalized_to_pixel(boxes, width, height):
    pixel_boxes = []
    for box in boxes:
        x0 = box[0] * width
        y0 = box[1] * height
        x1 = box[2] * width
        y1 = box[3] * height
        if [x0, y0, x1, y1] not in pixel_boxes:
            pixel_boxes.append([x0, y0, x1, y1])
    return pixel_boxes


def draw_and_save_boxes(image_pil, boxes, output_path="output_with_boxes.png"):
    # Convert to RGB if not already
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image_pil)
    for box in boxes:
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=1, width=3)  # Draw boxes
    
    # Save the image
    image_pil.save(output_path)
    return output_path




def manipulation_mask_generation(input_file, output_video, dino_model, dino_transform, image_pil, tags, device, box_threshold=0.25, text_threshold=0.2):
    text_threshold=0.2
    inputs = dino_transform(images=image_pil, text=tags, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)
    W, H = image_pil.size

    # Try with initial thresholds.
    results = dino_transform.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    boxes = results[0]["boxes"].cpu().numpy()
    boxes = convert_normalized_to_pixel(boxes, W, H)
    centers, labels = compute_box_centers(boxes)

    temp_dir = tempfile.gettempdir()
    temp_tensor_location = os.path.join(temp_dir, f"{os.path.basename(input_file[:-4])}_temp_tensor.pt")
    temp_vid_location = os.path.join(temp_dir, f"{os.path.basename(input_file[:-4])}_temp_video.mp4")

    cmd = [
        "python", "cosmos-transfer2.5/cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py",
        "--input_video", f"{input_file}",
        "--output_video", f"{temp_vid_location}",
        "--output_tensor", f"{temp_tensor_location}",
        "--mode", "points",
        "--prompt", f"{centers}",
        "--labels", f"{labels}"
    ]

    subprocess.run(cmd, check=True)

    ffmpeg_cmd = f'''
    ffmpeg -y -i "{temp_vid_location}" \
        -vf "format=gray,lut='if(gt(val\,0)\,255\,0)'" \
        -c:v libx264 -pix_fmt yuv420p "{output_video}"
    '''
    process = subprocess.run(
        shlex.split(ffmpeg_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    
