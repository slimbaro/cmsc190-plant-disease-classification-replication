"""
Include steps of Lab ImageSplit -> CLAHE (L) -> Channel Merge (L,A,B)

Input:  imagepreprocessing/bgr2lab/color
Output: imagepreprocessing/clahe_merged/color

NOTE: the paper does not state the hyperparameters for CLAHE (clip limit, tile grid size)
We used CLIP_LIMIT = 2.0 and TILE_GRID_SIZE = (8,8) as they are OpenCV's common defaults
"""

#pip install opencv-python tqdm (if not yet installed)

import os # filesystem operations
import cv2 # OpenCV for image processing
from pathlib import Path # path manipulation
from concurrent.futures import ThreadPoolExecutor # multi-threading
from tqdm import tqdm # progress bar

# paths for input and output
SCRIPT_DIR = Path(__file__).resolve().parent

input_dir = SCRIPT_DIR / "bgr2lab" / "color"
output_dir = SCRIPT_DIR / "clahe_merged" / "color"

output_dir.mkdir(parents = True, exist_ok = True) # create output directory if it doesn't exist
# parents = True -> creates any missing parent folders if nonexistent
# exist_ok = True -> no error if folder already exists

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# CLAHE hyperparameters (used default values)
# https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/ stated default values
CLIP_LIMIT = 40.0 # total contrast enhancement
TILE_GRID_SIZE = (8, 8) # divides image into 8x8 tiles to process each tile separately (for local contrast enhancement)

def apply_clahe_and_merge(args):
    """GOAL is to split LAB image, apply CLAHE, and merge back"""
    src_file, src_dir, dst_dir = args # get values from task tuple

    try: # preserve directory structure
        
        relative_path = src_file.relative_to(src_dir) # get file path relative to input folder (e.g., bgr2lab/color/Apple___Apple_scab/0a1b2c3d4e5f6g7h8i9j.jpg -> Apple___Apple_scab/0a1b2c3d4e5f6g7h8i9j.jpg)
        dst_file = dst_dir / relative_path # create matching output path (e.g., clahe_merged/color/Apple___Apple_scab/0a1b2c3d4e5f6g7h8i9j.jpg)
        dst_file.parent.mkdir(parents = True, exist_ok = True) # create output subfolders if needed

        # skip if already processed
        if dst_file.exists():
            return True

        img_lab = cv2.imread(str(src_file)) # read LAB image (from last step)

        # continue only if image was loaded successfully
        if img_lab is not None: 
            # 1st step: Split into L, A, B channels
            l_channel, a_channel, b_channel = cv2.split(img_lab)

            # 2nd step: Create CLAHE object
            clahe = cv2.createCLAHE(
                clipLimit = CLIP_LIMIT, 
                tileGridSize = TILE_GRID_SIZE
            )

            # Apply CLAHE to the lightness channel
            l_channel_clahe = clahe.apply(l_channel)

            # 3rd step: Merge channels back
            merged_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))

            # save processed image
            cv2.imwrite(str(dst_file), merged_lab)

            return True
        
    except Exception as e:
        print(f"Error on {src_file}: {e}") # print error if image failed to process
    
    return False


def main():
    # check if input folder exists
    if not input_dir.exists(): 
        print(f"Error: Path '{input_dir}' does not exist.")
        return

    # find image files inside input folder
    print("Gathering LAB images...")
    all_files = [
        Path(root) / file
        for root, _, files in os.walk(input_dir)
        for file in files
        if os.path.splitext(file)[1] in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(all_files)} images. Applying CLAHE to L-channel and merging...")

    # Create processing tasks
    tasks = [
        (f, input_dir, output_dir) 
        for f in all_files
    ]

    # process images using multiple threads for faster procesing
    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    apply_clahe_and_merge, 
                    tasks
                ), 
                total = len(tasks), 
                desc = "CLAHE + merge"
            )
        )

    print(f"\nDone! CLAHE-enhanced LAB dataset saved to:\n{output_dir}")

# start program
if __name__ == "__main__":
    main()