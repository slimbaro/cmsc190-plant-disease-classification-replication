"""
Scale existing .npz numpy array files to [-1, 1] (note that the result is still in .npz)

Input:  imagepreprocessing/numpy_arrays/color (uint8 arrays)
Output: imagepreprocessing/scaled_numpy_arrays/color (float32 arrays scaled to [-1, 1])
"""

import os  # filesystem operations
import numpy as np  # array operations
from pathlib import Path  # path manipulation
from concurrent.futures import ThreadPoolExecutor  # multi-threading
from tqdm import tqdm  # progress bar

# paths for input (.npz directory) and output
SCRIPT_DIR = Path(__file__).resolve().parent

input_dir = SCRIPT_DIR / "numpy_arrays" / "color"
output_dir = SCRIPT_DIR / "scaled" / "color"

output_dir.mkdir(parents=True, exist_ok=True)  # create output directory if it doesn't exist

MAX_FILES = 10_000

def scale_numpy_file(args):
    src_file, src_dir, dst_dir = args  # get values from task tuple

    try:
        relative_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        if dst_file.exists():
            return True

        # load the existing .npz file
        with np.load(src_file) as data:
            raw_array = data["image"]  # (H, W, 3) uint8 array in [0, 255]
        # scale NumPy array to float32 [-1, 1]
        scaled_array = (raw_array.astype(np.float32) / 127.5) - 1.0
        # save scaled array back to .npz format
        np.savez_compressed(dst_file, image=scaled_array)
        return True

    except Exception as e:
        print(f"Error on {src_file}: {e}")  # error handling

    return False

def main():
    if not input_dir.exists():  # check if input directory exists
        print(f"Error: Path '{input_dir}' does not exist.")
        return

    print("Gathering .npz files...")
    all_files = [  # collect .npz files
        Path(root) / file
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.endswith(".npz")
    ]

    unprocessed_files = [
        f for f in all_files
        if not (output_dir / f.relative_to(input_dir)).exists()
    ]

    files_to_process = unprocessed_files[:MAX_FILES]

    print(f"Found {len(all_files)} .npz files.")
    print(f"Scaling {len(files_to_process)} files to [-1, 1] range...")

    tasks = [(f, input_dir, output_dir) for f in files_to_process]  # prepare tasks

    with ThreadPoolExecutor() as executor:  # multi-threading
        list(
            tqdm(
                executor.map(scale_numpy_file, tasks),
                total=len(tasks),
                desc="Scaling NumPy arrays",
            )
        )
    print(f"\nDone! Scaled NumPy array dataset saved to:\n{output_dir}") # success message
if __name__ == "__main__":
    main()