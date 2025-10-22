import subprocess, pathlib

def run_sam2_pipeline(input_file, prompt_items, output_video):
    cmd = [
            "python", "cosmos-transfer2.5/cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py",
            "--input_video", f"{input_file}",
            "--output_video", f"{output_video}",
            "--mode", "prompt",
            "--prompt", f"{prompt_items}",
            "--visualize"
        ]

    subprocess.run(cmd, check=True)
