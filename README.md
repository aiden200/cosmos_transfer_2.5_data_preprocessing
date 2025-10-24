# Video Pre-Compute Pipeline (RAM tags / SAM2 seg / Edges / Masks / Prompts / Filtered Edges)

Batch-prepare per-video controls for downstream video editing or generation:
- RAM (tags/labels) -> labels.txt
- SAM2 (semantic segmentation video) -> seg.mp4
- Edges (edge video) -> edge.mp4
- Masks (object masks from text) -> mask.mp4
- Prompt (scene description) -> prompt.txt
- Filtered Edges (masking out edges) -> filtered.mp4

Outputs are written under `pipeline/outputs/<video_basename>/`.


## Features
- Load inputs from a folder or a newline-separated file list
- Choose any combination of controls: ram, sam2, edge, mask, prompt
- Skip already-generated artifacts with --skip
- GPU/CPU selectable device; configurable batch + image size for RAM

## Installation
Cosmos transfer 2.5
```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5
uv sync
source .venv/bin/activate
```

This repository
```
cd ..
git clone https://github.com/aiden200/cosmos_transfer_2.5_data_preprocessing.git pipeline
```

## RAM Installation
```bash
uv pip install fairscale
uv pip install ultralytics
cd pipeline
git clone https://github.com/xinyu1205/recognize-anything.git recognize_anything
uv pip install -r ./recognize_anything/requirements.txt
uv pip install -e ./recognize_anything/
cd ..
```


## Directory Layout
```
pipeline/
  inputs/
    handshake_1.mp4
    fistbump_1.mp4
  outputs/
    handshake_1/
      labels.txt      # if ram
      seg.mp4         # if sam2
      edge.mp4        # if edge
      mask.mp4        # if mask
      prompt.txt      # if prompt
      filtered.mp4    # if mask_edges
    fistbump_1/
      ...
```

## Quick Start Examples

### 1) From a folder: RAM + Edges
```bash
python -m pipeline.full_pipeline \
  --video-load-type input_folder \
  --input-folder pipeline/inputs \
  --control-nets ram,edge \
  --ram-checkpoint checkpoints/ram_vit_b.pth \
  --device cuda \
  --batch-size 16 \
  --img-size 384 \
  --skip
```
Generates labels.txt and edge.mp4 per input video (skips ones that already exist).

### 2) Masks with object list
```bash
python -m pipeline.full_pipeline \
  --video-load-type input_folder \
  --input-folder pipeline/inputs \
  --control-nets mask \
  --mask-prompt "person" \
  --device cuda \
  --skip
```
Generates mask.mp4 where only the object is kept (others suppressed).

### 3) Full pipeline (RAM + SAM2 + Edge + Mask + Prompt + Filtered Edge)
```bash
python -m pipeline.full_pipeline \
  --video-load-type input_folder \
  --input-folder pipeline/inputs \
  --control-nets edge \
  --ram-checkpoint cosmos-transfer2.5/Grounded-Segment-Anything/ram_swin_large_14m.pth \
  --mask-prompt "person" \
  --device cuda \
  --verbose
```

## CLI Overview
```
--video-load-type    input_folder | input_file   (required)
--input-folder       <dir>                       (required for input_folder)
--input-file         <file.txt>                  (required for input_file)
--control-nets       ram,sam2,edge,mask,prompt   (comma-separated; required)
--mask-prompt        "obj1,obj2,..."             (required if mask selected)
--ram-checkpoint     <path/to/ram.ckpt>          (required if ram selected)
--device             cuda | cuda:0 | cpu         (default: cuda)
--batch-size         <int>                        (default: 16; for RAM)
--img-size           <int>                        (default: 384; for RAM)
--skip               (flag)                       skip already-generated files
--extensions         mp4,mov,avi,...              (default: common formats)
--verbose            (flag)                       extra logs
```

## Input Options
### Load from folder
```bash
--video-load-type input_folder --input-folder pipeline/inputs
```
Uses files with allowed --extensions (default includes mp4,mov,avi,mkv,m4v,webm).

### Load from file list
```bash
--video-load-type input_file --input-file videos.txt
```
Each line is a path to a video.

## Troubleshooting
- “Unknown control(s)”: Check spelling; allowed set is {ram, sam2, edge, mask, prompt}.
- “Need to specify some mask objects”: Add --mask-prompt "a,b,c" when mask is included.
- “You included ‘ram’… but did not provide –ram-checkpoint.”: Supply a valid checkpoint path.
- No videos found: Confirm --input-folder and --extensions, or check paths in your file list.
