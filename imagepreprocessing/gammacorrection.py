"""
Gamma Correction applied to Adaptive Median Filter (AMF) output dataset

Input:  imagepreprocessing/amf/color
Output: imagepreprocessing/gammacorrection/color
"""

import os  # filesystem operations
import cv2  # OpenCV for image reading/writing
import numpy as np  # array operations
from pathlib import Path  # path manipulation
from concurrent.futures import ThreadPoolExecutor  # multi-threading
from tqdm import tqdm  # progress bar

# paths for input and output
SCRIPT_DIR = Path(__file__).resolve().parent

input_dir = SCRIPT_DIR / "amf" / "color"
output_dir = SCRIPT_DIR / "gammacorrection" / "color"

output_dir.mkdir(parents=True, exist_ok=True)  # create output directory if it doesn't exist

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# Gamma Correction Hyperparameter (10.3389/frai.2026.1751118: DOI of a study that used PlantVillage set and used 1.2 as their gamma value)
GAMMA = 1.2  # Gamma > 1.0 brightens; Gamma < 1.0 darkens


def build_gamma_lut(gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table

# Pre-compute Lookup Table for efficiency
GAMMA_LUT = build_gamma_lut(GAMMA)

def apply_gamma_correction(args):
    src_file, src_dir, dst_dir = args 

    try:
        # output path
        relative_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists():  # don't re-process if already exists
            return True

        img_bgr = cv2.imread(str(src_file))  # read AMF image

        if img_bgr is not None:
            # apply gamma correction using LUT mapping
            img_gamma = cv2.LUT(img_bgr, GAMMA_LUT)

            cv2.imwrite(str(dst_file), img_gamma)  # save image
            return True

    except Exception as e:
        print(f"Error on {src_file}: {e}")  # error handling

    return False


def main():
    if not input_dir.exists():  # check if input directory exists
        print(f"Error: Path '{input_dir}' does not exist.")
        return

    print("Gathering AMF-filtered images...")
    all_files = [  # collect images
        Path(root) / file
        for root, _, files in os.walk(input_dir)
        for file in files
        if os.path.splitext(file)[1] in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(all_files)} images. Applying Gamma Correction (gamma={GAMMA})...")

    tasks = [(f, input_dir, output_dir) for f in all_files]  # prepare tasks for multi-threading

    with ThreadPoolExecutor() as executor:  # multi-threading for faster processing
        list(
            tqdm(
                executor.map(apply_gamma_correction, tasks),
                total=len(tasks),
                desc="Gamma Correction",
            )
        )

    print(f"\nDone! Gamma-corrected dataset saved to:\n{output_dir}")  # success message


if __name__ == "__main__":
    main()