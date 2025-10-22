import os
import cv2
import torch
from PIL import Image

from pipeline.utils.filepath_functions import create_folders, get_filename_no_suffix, find_video_tags
from pipeline.control_net_generation.get_object_labels import retrieve_tags
from pipeline.control_net_generation.segment_videos import run_sam2_pipeline
from pipeline.control_net_generation.get_object_edges import generate_edges
from pipeline.control_net_generation.get_object_depth import get_depth
from pipeline.control_net_generation.get_prompt import get_prompt
from pipeline.control_net_generation.mask_videos import run_mask_pipeline
import argparse, os, sys

#TODO: Depth and prompt


def _split_csv(s):
    # helper: turn "a,b, c" -> ["a","b","c"], allow empty
    if s is None: return None
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute controls (RAM tags, SAM2 seg, edges, masks, prompts) for videos."
    )

    parser.add_argument(
        "--video-load-type",
        choices=["input_folder", "input_file"],
        required=True,
        help="Load videos from a folder or from a text file of absolute/relative paths."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        help="Folder containing input videos (used when --video-load-type input_folder)."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Text file with one video filepath per line (used when --video-load-type input_file)."
    )

    parser.add_argument(
        "--control-nets",
        type=_split_csv,
        required=True,
        help="Comma-separated list of controls to generate. Options: ram,sam2,edge,mask,prompt"
    )

    parser.add_argument(
        "--mask-prompt",
        type=_split_csv,
        help="Comma-separated list of object names to keep in the mask (required if 'mask' in control-nets)."
    )

    parser.add_argument(
        "--ram-checkpoint",
        type=str,
        default=None,
        help="Path to RAM checkpoint (required if 'ram' in control-nets)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for models (e.g., 'cuda', 'cuda:0', or 'cpu')."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used by RAM tag extraction."
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=384,
        help="Image size used by RAM transform/inference."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging for preprocessing steps."
    )

    # Behavior
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip videos that already have the corresponding outputs."
    )
    parser.add_argument(
        "--extensions",
        type=_split_csv,
        default="mp4,mov,avi,mkv,m4v,webm",
        help="Comma-separated list of video extensions to include when loading from a folder."
    )

    args = parser.parse_args()



    allowed_controls = {"ram", "sam2", "edge", "mask", "prompt"}
    control_nets = [c.lower() for c in (args.control_nets or [])]
    unknown = [c for c in control_nets if c not in allowed_controls]
    if unknown:
        raise ValueError(f"Unknown control(s): {unknown}. Allowed: {sorted(allowed_controls)}")
    args.control_nets = control_nets


    if args.video_load_type == "input_folder":
        if not args.input_folder:
            raise ValueError("--input-folder is required when --video-load-type input_folder")
        args.input_folder = os.path.abspath(os.path.expanduser(args.input_folder))
        if not os.path.isdir(args.input_folder):
            raise FileNotFoundError(f"Input folder not found: {args.input_folder}")

        # args.extensions is already a list because type=_split_csv was used
        exts_list = args.extensions or ["mp4", "mov", "avi", "mkv", "m4v", "webm"]
        exts = {"." + e.lower().lstrip(".") for e in exts_list}

        _folder_paths = []
        for f in os.listdir(args.input_folder):
            fp = os.path.join(args.input_folder, f)
            if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in exts:
                _folder_paths.append(fp)
    else: 
        if not args.input_file:
            raise ValueError("--input-file is required when --video-load-type input_file")
        args.input_file = os.path.abspath(os.path.expanduser(args.input_file))
        if not os.path.isfile(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

    if "mask" in args.control_nets and not args.mask_prompt:
        raise ValueError("You included 'mask' in --control-nets but did not provide --mask-prompt.")

    if "ram" in args.control_nets and not args.ram_checkpoint:
        raise ValueError("You included 'ram' in --control-nets but did not provide --ram-checkpoint.")


    ram_checkpoint_path = args.ram_checkpoint
    ram_img_size = args.img_size
    ram_device = args.device
    ram_batch_size = args.batch_size

    video_paths = []

    if args.video_load_type == "input_folder":
        print(_folder_paths)
        video_paths=_folder_paths
        # video_paths = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f))]
    else:
        # Load from a file specifying filepaths
        args.input_file
        with open(args.input_file, "r") as file:
            for line in file:
                video_paths.append(line.strip())

    # get tags to load. ["ram", "sam2", "edge", "depth", "mask"]
    control_nets = args.control_nets

    # Skip already loaded videos
    skip_already_loaded = args.skip

    # Create folders
    create_folders(video_paths)

    # Tags 
    tags_with_labels = {}

    if "ram" in control_nets:
        if not skip_already_loaded:
            ram_video_files = find_video_tags(video_paths, "labels.txt")
        else:
            ram_video_files = video_paths
        
        print(f"Generating {len(ram_video_files)} RAM (tags) files")

        tags = retrieve_tags(ram_checkpoint_path, ram_video_files, device="cuda", batch_size=16, img_size=384, verbose=False)
        
        print(tags)
        print(ram_video_files)
        ## TODO: Fix
        if len(tags) != len(ram_video_files):
            raise ValueError(f"tags length ({len(tags)}) != files length ({len(ram_video_files)})")

        for video_path, tag in zip(ram_video_files, tags):
            basename = get_filename_no_suffix(video_path)
            parent_dir = os.path.join("pipeline/outputs", basename)
            labels_path = os.path.join(parent_dir, "labels.txt")
            with open(labels_path, "w") as f:
                f.write(tag)

        for video_path in ram_video_files:
            basename = get_filename_no_suffix(video_path)
            labels_path = os.path.join("pipeline/outputs", basename, "labels.txt")
            with open(labels_path, "r") as f:
                tags_with_labels[basename] = f.read()


        for video_path, tag in zip(ram_video_files, tags):
            basename = get_filename_no_suffix(video_path)
            parent_dir = os.path.join("pipeline/outputs", basename)
            labels_path = os.path.join(parent_dir, "labels.txt")
            with open(labels_path, "w") as f:
                f.write(tag)

        for video_path in ram_video_files:
            basename = get_filename_no_suffix(video_path)
            labels_path = os.path.join("pipeline/outputs", basename, "labels.txt")
            with open(labels_path, "r") as f:
                tags_with_labels[basename] = f.read()
        

    
    # Segmented videos
    if "sam2" in control_nets:
        if not skip_already_loaded:
            sam_video_files = find_video_tags(video_paths, "seg.mp4")
        else:
            sam_video_files = video_paths

        print(f"Generating {len(sam_video_files)} SAM2 (segment) files")
        for video_path in sam_video_files:
            basename = get_filename_no_suffix(video_path)
            sam_video_path = os.path.join("pipeline/outputs", basename, "seg.mp4")
            run_sam2_pipeline(video_path, tags_with_labels[basename], sam_video_path)


    # Get edges
    if "edge" in control_nets:
        if not skip_already_loaded:
            edge_video_files = find_video_tags(video_paths, "edge.mp4")
        else:
            edge_video_files = video_paths
        print(f"Generating {len(edge_video_files)} edge files")
        
        for video_path in edge_video_files:
            basename = get_filename_no_suffix(video_path)
            edge_video_path = os.path.join("pipeline/outputs", basename, "edge.mp4")
            generate_edges(video_path, edge_video_path)


    # Get masks: comma separated objects
    if "mask" in control_nets:
        assert args.mask_prompt, "Need to specify some mask objects"

        mask_objects = args.mask_prompt
        
        if not skip_already_loaded:
            mask_video_files = find_video_tags(video_paths, "mask.mp4")
        else:
            mask_video_files = video_paths
        print(f"Generating {len(mask_video_files)} mask files")

        for video_path in mask_video_files:
            basename = get_filename_no_suffix(video_path)
            mask_video_path = os.path.join("pipeline/outputs", basename, "mask.mp4")
            run_mask_pipeline(video_path, mask_objects, mask_video_path)

    # Get prompt
    if "prompt" in control_nets:
        prompts = {}
        if not skip_already_loaded:
            prompt_files = find_video_tags(video_paths, "prompt.txt")
        else:
            prompt_files = video_paths
        
        print(f"Generating {len(prompt_files)} prompt files")
        for video_path in prompt_files:
            basename = get_filename_no_suffix(video_path)
            prompt = get_prompt(video_path)
            #TODO
            prompt_text_path = os.path.join("pipeline/outputs", basename, "prompt.txt")
            with open(prompt_text_path, "w") as f:
                f.write(prompt)

        for video_path in prompt_files:
            basename = get_filename_no_suffix(video_path)
            prompt_text_path = os.path.join("pipeline/outputs", basename, "prompt.txt")
            with open(prompt_text_path, "r") as f:
                prompts[basename] = f.read()





if __name__ == "__main__":
    main()


'''
python3 cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
    --output_video pipeline/outputs/manipulation_1/output_video.mp4 \
    --mode prompt \
    --prompt "computer, table, equipment, person, man, office, office supply, room, stand, job, yellow, potato chip bag, cup, cigarette, glass tube" \
    --input_video pipeline/inputs/episode_1.mp4
'''
