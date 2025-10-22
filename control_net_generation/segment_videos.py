from cosmos_transfer2._src.transfer2.auxiliary.sam2.sam2_pipeline import main


def run_sam2_pipeline(input_file, prompt_items, output_video):
    args = [
        "--output_video", f"{output_video}",
        "--mode", "prompt",
        "--prompt", f"{prompt_items}",
        "--input_video", f"{input_file}",
    ]
    main(args)