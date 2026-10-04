"""
LAB2BGR is to convert image from LAB color space back to BGR

Input:  imagepreprocessing/clahe_merged/color
Output: imagepreprocessing/lab2bgr/color
"""

#pip install opencv-python tqdm (if not yet installed)

import os  # filesystem operations
import cv2  # OpenCV for image processing
from pathlib import Path  # path manipulation
from concurrent.futures import ThreadPoolExecutor  # multi-threading
from tqdm import tqdm  # progress bar

# paths for input and output
SCRIPT_DIR = Path(__file__).resolve().parent

input_dir = SCRIPT_DIR / "clahe_merged" / "color"
output_dir = SCRIPT_DIR / "lab2bgr" / "color"

output_dir.mkdir(parents=True, exist_ok=True)  # create output directory if it doesn't exist

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def convert_lab_to_bgr(args):
    """GOAL is to read CLAHE-enhanced LAB image and convert it back to BGR"""
    src_file, src_dir, dst_dir = args  # get values from task tuple

    try:
        # preserve directory structure
        relative_path = src_file.relative_to(src_dir)  # e.g. Apple___Apple_scab/img001.jpg
        dst_file = dst_dir / relative_path  # matching output path
        dst_file.parent.mkdir(parents=True, exist_ok=True)  # create output subfolders if needed

        # skip if already processed
        if dst_file.exists():
            return True

        img_lab = cv2.imread(str(src_file))  # read LAB image (from clahe_merge)

        if img_lab is not None:
            # convert LAB color space back to BGR
            img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

            # save converted image
            cv2.imwrite(str(dst_file), img_bgr)
            return True

    except Exception as e:
        print(f"Error on {src_file}: {e}")  # print error if image failed to process

    return False


def main():
    # check if input folder exists
    if not input_dir.exists():
        print(f"Error: Path '{input_dir}' does not exist.")
        return

    # find image files inside input folder
    print("Gathering CLAHE-merged LAB images...")
    all_files = [
        Path(root) / file
        for root, _, files in os.walk(input_dir)
        for file in files
        if os.path.splitext(file)[1] in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(all_files)} images. Converting LAB -> BGR...")

    # create processing tasks
    tasks = [(f, input_dir, output_dir) for f in all_files]

    # process images using multiple threads for faster processing
    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(convert_lab_to_bgr, tasks),
                total=len(tasks),
                desc="LAB2BGR",
            )
        )

    print(f"\nDone! BGR-converted dataset saved to:\n{output_dir}")


# start program
if __name__ == "__main__":
    main()