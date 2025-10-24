import os

import numpy as np

import torch
import cv2
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO


def get_first_frame_pil(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image_pil = Image.fromarray(frame_rgb)

    return image_pil


def convert_normalized_to_pixel(boxes, width, height):
    pixel_boxes = []
    for box in boxes:
        x0 = box[0] * width
        y0 = box[1] * height
        x1 = box[2] * width
        y1 = box[3] * height
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



def load_dino_model(grounding_dino_path="IDEA-Research/grounding-dino-base", device="cuda"):

    processor = AutoProcessor.from_pretrained(grounding_dino_path)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_path).to(device)

    return grounding_model, processor


def load_yolo_model(model_path="yolov9c.pt"):

    return YOLO(model_path)


def yolo_largest_person_box(yolo_model, image_pil, conf=0.90, device="cuda"):


    results = yolo_model([image_pil])
    results[0].save(filename="result.jpg")



    if not results or len(results) == 0:
        return None

    r = results[0]
    if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
        return None

    boxes_xyxy = r.boxes.xyxy.detach().cpu().numpy()  # (N, 4) in pixels
    classes = r.boxes.cls.detach().cpu().numpy().astype(int)  # (N,)

    person_mask = (classes == 0)
    person_boxes = boxes_xyxy[person_mask]

    if person_boxes.size == 0:
        return None


    # Pick the largest by area
    areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1])
    idx = int(np.argmax(areas))
    x0, y0, x1, y1 = person_boxes[idx].tolist()
    largest_box_str = f"{x0},{y0},{x1},{y1}"

    # Also return all person boxes as a Python list for optional drawing
    all_person_boxes = person_boxes.tolist()
    return largest_box_str




def return_largest_bounding_box(dino_model, dino_transform, image_pil, tags, device, box_threshold=0.25, text_threshold=0.2):
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

    max_size = 0
    boxes_coords = None
    for box in boxes:
        x0, y0, x1, y1 = box
        area = (x1-x0) / 2 + (y1 - y0) / 2
        if area > max_size:
            max_size = area
            boxes_coords = f"{x0},{y0},{x1},{y1}"

    return boxes_coords
