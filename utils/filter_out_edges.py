#!/usr/bin/env python3
"""
Apply a (grown/feathered) mask video to an edge video:
- Keep edge pixels only where (expanded) mask passes threshold.
- Optional: fill tiny holes (closing), expand (dilate), and feather.

Usage (MP4 output):
  python apply_mask.py \
    --threshold 0 \
    --mask_grow_px 3 \
    --mask_close_px 3 \
    --feather_px 2

Optional: PNG sequence (RGBA):
  python apply_mask.py --png_dir out_pngs
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

def ensure_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def make_kernel(px):
    """Return an odd-sized elliptical kernel roughly px radius."""
    k = max(1, int(px) * 2 + 1)  # ensure odd
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def grow_and_feather_mask(mask_frame, threshold, grow_px=0, close_px=0, feather_px=0):
    """
    1) Binarize by threshold.
    2) Optional closing (fill small holes/gaps).
    3) Optional dilation (expand outward by ~grow_px).
    4) Optional feather: soft ramp from 0..255 near mask boundary.
    Returns uint8 mask (0..255).
    """
    mask_gray = ensure_gray(mask_frame)

    # Binarize
    _, mask_bin = cv2.threshold(mask_gray, threshold, 255, cv2.THRESH_BINARY)

    # Close (fills pinholes/small gaps inside the mask)
    if close_px and close_px > 0:
        k_close = make_kernel(close_px)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k_close, iterations=1)

    # Dilate (expand outward)
    if grow_px and grow_px > 0:
        k_dilate = make_kernel(grow_px)
        mask_bin = cv2.dilate(mask_bin, k_dilate, iterations=1)

    # Feather: create a soft band around the boundary that passes some alpha
    if feather_px and feather_px > 0:
        # Distance from background to mask (in pixels)
        inv = cv2.bitwise_not(mask_bin)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)  # float32
        # Within feather band, ramp alpha up to 255
        # Pixels already in mask_bin stay at 255.
        feather_band = np.clip((feather_px - dist) / max(1e-6, feather_px), 0.0, 1.0)
        feather_alpha = (feather_band * 255).astype(np.uint8)
        # Combine: max keeps full mask at 255, boundary gets 1..254
        mask_soft = np.maximum(mask_bin, feather_alpha)
        return mask_soft

    return mask_bin

def apply_mask(edge_frame_bgr, mask_frame, threshold, grow_px=0, close_px=0, feather_px=0):
    """
    Apply a grown/feathered mask to edge_frame_bgr.
    Returns (BGR, RGBA).
    """
    h, w = edge_frame_bgr.shape[:2]
    if mask_frame.shape[:2] != (h, w):
        mask_frame = cv2.resize(mask_frame, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = grow_and_feather_mask(
        mask_frame,
        threshold=threshold,
        grow_px=grow_px,
        close_px=close_px,
        feather_px=feather_px,
    )

    # Bitwise AND uses mask as 0..255 alpha on each channel
    out_bgr = cv2.bitwise_and(edge_frame_bgr, edge_frame_bgr, mask=mask)

    # RGBA with provided alpha
    out_rgba = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2BGRA)
    out_rgba[:, :, 3] = mask  # soft alpha if feathering used

    return out_bgr, out_rgba

def filter_out_edges(edges_p, mask_p, out_p, threshold=0, mask_grow_px=0, mask_close_px=0, feather_px=0):

    edge_cap = cv2.VideoCapture(str(edges_p))
    mask_cap = cv2.VideoCapture(str(mask_p))

    if not edge_cap.isOpened():
        raise RuntimeError(f"Failed to open edge video: {edges_p}")
    if not mask_cap.isOpened():
        raise RuntimeError(f"Failed to open mask video: {mask_p}")

    fps = edge_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(edge_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(edge_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try a different --fourcc (e.g., avc1, XVID, MJPG).")

    last_edge = last_mask = None

    while True:
        edge_ok, edge_frame = edge_cap.read()
        mask_ok, mask_frame = mask_cap.read()

        if edge_frame is None or mask_frame is None:
            break

        last_edge = edge_frame
        last_mask = mask_frame

        # Ensure edge is BGR
        if edge_frame.ndim == 2:
            edge_frame = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)

        out_bgr, out_rgba = apply_mask(
            edge_frame,
            mask_frame,
            threshold=threshold,
            grow_px=mask_grow_px,
            close_px=mask_close_px,
            feather_px=feather_px,
        )
        writer.write(out_bgr)


    edge_cap.release()
    mask_cap.release()
    writer.release()
