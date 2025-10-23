import subprocess, pathlib
import tempfile, os

def run_sam2_pipeline(input_file, prompt_items, output_video, box=False):
    temp_dir = tempfile.gettempdir()
    temp_tensor_location = os.path.join(temp_dir, f"{os.path.basename(input_file[:-4])}_temp_tensor.pt")

    if box:
        cmd = [
                "python", "cosmos-transfer2.5/cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py",
                "--input_video", f"{input_file}",
                "--output_video", f"{output_video}",
                "--output_tensor", f"{temp_tensor_location}",
                "--mode", "box",
                "--box", f"{prompt_items}"
            ]
    else:
        cmd = [
                "python", "cosmos-transfer2.5/cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py",
                "--input_video", f"{input_file}",
                "--output_video", f"{output_video}",
                "--output_tensor", f"{temp_tensor_location}",
                "--mode", "prompt",
                "--prompt", f"{prompt_items}"
            ]

    subprocess.run(cmd, check=True)
