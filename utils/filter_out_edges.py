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

def main():
    ap = argparse.ArgumentParser(description="AND an edge video with a (grown/feathered) mask video.")
    # If you want CLI inputs again, uncomment these:
    # ap.add_argument("--edges", required=True, help="Path to edge video (e.g., edges.mp4)")
    # ap.add_argument("--mask", required=True, help="Path to mask video (e.g., mask.mp4)")
    # ap.add_argument("--out", required=True, help="Output video path (e.g., masked_edges.mp4)")

    ap.add_argument("--threshold", type=int, default=0,
                    help="Mask binarization threshold [0..255], keep where mask>threshold. Default 0.")
    ap.add_argument("--mask_grow_px", type=int, default=0,
                    help="Grow/dilate mask outward by ~N pixels (ellipse kernel).")
    ap.add_argument("--mask_close_px", type=int, default=0,
                    help="Close small holes/gaps inside mask with ~N pixel kernel.")
    ap.add_argument("--feather_px", type=int, default=0,
                    help="Feather edge by ~N pixels (soft alpha near boundary).")
    ap.add_argument("--fourcc", default="mp4v",
                    help="OpenCV FOURCC (mp4v, avc1, XVID, MJPG). Default mp4v.")
    ap.add_argument("--png_dir", default=None,
                    help="Optional: directory to also dump RGBA PNG frames with transparency.")
    ap.add_argument("--stop_at", choices=["shortest", "edges", "mask"], default="shortest",
                    help="When streams differ in length, stop at: shortest (default) | edges | mask.")
    args = ap.parse_args()

    # Hardcoded paths (as in your snippet). Re-enable CLI above if desired.
    edges_p = "/home/nvidia/workspace/transfer-guide/cosmos-transfer2.5/assets/xpeng/manipulation_2_edge.mp4"
    mask_p  = "/home/nvidia/workspace/transfer-guide/cosmos-transfer2.5/Grounded-Segment-Anything/outputs/cosmos/episode_1.mp4"
    out_p   = "/home/nvidia/workspace/transfer-guide/cosmos-transfer2.5/assets/xpeng/manipulation_2_masked_edges.mp4"

    edge_cap = cv2.VideoCapture(str(edges_p))
    mask_cap = cv2.VideoCapture(str(mask_p))

    if not edge_cap.isOpened():
        raise RuntimeError(f"Failed to open edge video: {edges_p}")
    if not mask_cap.isOpened():
        raise RuntimeError(f"Failed to open mask video: {mask_p}")

    fps = edge_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(edge_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(edge_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try a different --fourcc (e.g., avc1, XVID, MJPG).")

    png_dir = None
    if args.png_dir:
        png_dir = Path(args.png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    frames_written = 0
    last_edge = last_mask = None

    while True:
        edge_ok, edge_frame = edge_cap.read()
        mask_ok, mask_frame = mask_cap.read()

        if not edge_ok:
            if args.stop_at == "edges" and mask_ok:
                break
            elif args.stop_at == "mask" and mask_ok and last_edge is not None:
                edge_frame = last_edge
            else:
                if args.stop_at == "shortest":
                    break
        if not mask_ok:
            if args.stop_at == "mask" and edge_ok:
                break
            elif args.stop_at == "edges" and edge_ok and last_mask is not None:
                mask_frame = last_mask
            else:
                if args.stop_at == "shortest":
                    break

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
            threshold=args.threshold,
            grow_px=args.mask_grow_px,
            close_px=args.mask_close_px,
            feather_px=args.feather_px,
        )
        writer.write(out_bgr)

        if png_dir is not None:
            cv2.imwrite(str(png_dir / f"frame_{frames_written:06d}.png"), out_rgba)

        frames_written += 1

    edge_cap.release()
    mask_cap.release()
    writer.release()
    print(f"Wrote {frames_written} frames to {out_p}")
    if png_dir is not None:
        print(f"Also wrote RGBA PNGs to: {png_dir}")

if __name__ == "__main__":
    main()