from recognize_anything.ram.models import ram
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
    return ram_model


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def retrieve_tags(
    ram_checkpoint_path,
    video_paths,
    device="cuda",
    batch_size=16,
    img_size=384,
    verbose=False,
):
    ram_model = initialize_ram_model(ram_checkpoint_path, img_size, device)
    ram_transform = get_ram_transform(img_size)

    pil_images = []
    keep_idx = []
    for idx, vp in enumerate(video_paths):
        frame = get_first_frame(vp)
        if frame is None:
            if verbose:
                print(f"[WARN] Could not read first frame: {vp}")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
        keep_idx.append(idx)

    if not pil_images:
        return []

    tensors = [ram_transform(img) for img in pil_images]
    tags_out = []
    ram_model.to(device)
    ram_model.eval()

    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size], dim=0).to(device, non_blocking=True)
            responses = inference_ram(batch, ram_model)

            for res in responses:
                s = res[0] if isinstance(res, (list, tuple)) else res
                s = s.replace(' |', ',')
                tags_out.append(s)

    return tags_out