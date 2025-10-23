from pipeline.recognize_anything.ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS
import torch
import os
import cv2
import torch
from PIL import Image


def get_ram_transform(image_size=384):
    return TS.Compose([TS.Resize((image_size, image_size)), TS.ToTensor(), TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def initialize_ram_model(ram_checkpoint_path, image_size, device):
    print("Initializing RAM++ Model...")
    ram_model = ram(pretrained=ram_checkpoint_path, image_size=image_size, vit='swin_l').eval().to(device)
    ram_model.eval()
    return ram_model


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def retrieve_tags(
    ram_model,
    ram_transform,
    video_path,
    device="cuda"
):
    frame = get_first_frame(video_path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    ram_input = ram_transform(img).unsqueeze(0).to(device)
    res = inference_ram(ram_input, ram_model)
    s = res[0] if isinstance(res, (list, tuple)) else res
    s = s.replace(' |', ',')
    return s
    
