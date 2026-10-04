"""
Convert BGR image dataset to RGB format.

Input:  imagepreprocessing/gammacorrection/color
Output: imagepreprocessing/bgr2rgb/color
"""

import os  # filesystem operations
import cv2  # OpenCV for image reading/writing
import numpy as np  # array operations
from pathlib import Path  # path manipulation
from concurrent.futures import ThreadPoolExecutor  # multi-threading
from tqdm import tqdm  # progress bar

# paths for input and output
SCRIPT_DIR = Path(__file__).resolve().parent

input_dir = SCRIPT_DIR / "gammacorrection" / "color"
output_dir = SCRIPT_DIR / "bgr2rgb" / "color"

output_dir.mkdir(parents=True, exist_ok=True)  # create output directory if it doesn't exist

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def convert_bgr2rgb(args):
    src_file, src_dir, dst_dir = args  

    try:
        # output path
        relative_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists():  # skip if already processed
            return True

        img_bgr = cv2.imread(str(src_file))  # read BGR image

        if img_bgr is not None:
            # Convert BGR channel order to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Save the image (cv2.imwrite expects BGR, so cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) ensures the RGB channel orientation is preserved correctly when saved to disk)
            cv2.imwrite(str(dst_file), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            return True

    except Exception as e:
        print(f"Error on {src_file}: {e}")  # error handling

    return False


def main():
    if not input_dir.exists():  # check if input directory exists
        print(f"Error: Path '{input_dir}' does not exist.")
        return

    print("Gathering Gamma-corrected images...")
    all_files = [  # collect images
        Path(root) / file
        for root, _, files in os.walk(input_dir)
        for file in files
        if os.path.splitext(file)[1] in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(all_files)} images. Converting BGR to RGB...")

    tasks = [(f, input_dir, output_dir) for f in all_files]  # prepare tasks

    with ThreadPoolExecutor() as executor:  # multi-threading
        list(
            tqdm(
                executor.map(convert_bgr2rgb, tasks),
                total=len(tasks),
                desc="BGR2RGB",
            )
        )

    print(f"\nDone! BGR to RGB converted dataset saved to:\n{output_dir}")  # success message


if __name__ == "__main__":
    main()