"""
AMF or the "Adaptive median filter (AMF)" is used to remove impulse/salt-and-pepper noise 

Input:  imagepreprocessing/lab2bgr/color
Output: imagepreprocessing/amf/color
"""

#pip install opencv-python scipy tqdm (if not yet installed)

import os  # filesystem operations
import cv2  # OpenCV for image reading/writing
import numpy as np  # array operations
from scipy.ndimage import median_filter, minimum_filter, maximum_filter  # windowed filters
from pathlib import Path  # path manipulation
from concurrent.futures import ThreadPoolExecutor  # multi-threading
from tqdm import tqdm  # progress bar

# paths for input and output
SCRIPT_DIR = Path(__file__).resolve().parent

input_dir = SCRIPT_DIR / "lab2bgr" / "color"
output_dir = SCRIPT_DIR / "amf" / "color"

output_dir.mkdir(parents=True, exist_ok=True)  # create output directory if it doesn't exist

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# AMF hyperparameters
MIN_WINDOW_SIZE = 3  # starting window size (must be odd)
MAX_WINDOW_SIZE = 7  # largest window size (must be odd)


def adaptive_median_filter_channel(channel: np.ndarray) -> np.ndarray:
    """ Apply AMF to a single 2D channel. """
    result = channel.copy() # initialize result array
    finalized = np.zeros(channel.shape, dtype=bool)  # tracks which pixels are done

    window_size = MIN_WINDOW_SIZE # start with 3x3
    while window_size <= MAX_WINDOW_SIZE:
        # compute local median, min, max across whole channel
        local_median = median_filter(channel, size=window_size)
        local_min = minimum_filter(channel, size=window_size)
        local_max = maximum_filter(channel, size=window_size)

        # Stage A: checks is min < median < max
        # if true, median is considered "clean" and can be used
        # if false, it will be replaced with the local median from the next larger window
        stage_a_pass = (local_median > local_min) & (local_median < local_max)

        # Stage B: checks min < pixel < max
        # If true, pixel is considered "clean" and can be used
        # if false, it will be replaced with the local median from the next larger window
        stage_b_pass = (channel > local_min) & (channel < local_max)

        # pixels that pass Stage A AND haven't been finalized get resolved now
        resolve_now = stage_a_pass & (~finalized)

        keep_original = resolve_now & stage_b_pass # if original pixel is also clean (Stage B), keep it
        use_median = resolve_now & (~stage_b_pass) # if original pixel is noisy (Stage B), replace with median

        result[keep_original] = channel[keep_original] # save original pixel value
        result[use_median] = local_median[use_median] # save local median value

        finalized |= resolve_now # mark pixels as finalized

        window_size += 2  # grow window size (must stay odd) so 3 -> 5 -> 7

    # unresolved pixels 
    # if still unresolved after 7x7, use largest-window median
    result[~finalized] = local_median[~finalized]

    return result # return cleaned channel


def apply_amf(args):
    """GOAL is to read a BGR image and apply AMF independently to each of its 3 channels"""
    src_file, src_dir, dst_dir = args  # get values from task tuple

    try:
        # output path
        relative_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists(): # don't re-process if already exists
            return True

        img_bgr = cv2.imread(str(src_file))  # read BGR image (from lab2bgr)

        if img_bgr is not None:
            # apply AMF to each channel (B, G, R) separately
            b, g, r = cv2.split(img_bgr)
            b_filtered = adaptive_median_filter_channel(b)
            g_filtered = adaptive_median_filter_channel(g)
            r_filtered = adaptive_median_filter_channel(r)

            img_filtered = cv2.merge((b_filtered, g_filtered, r_filtered)) # combine filtered channels

            cv2.imwrite(str(dst_file), img_filtered) # save image
            return True

    except Exception as e:
        print(f"Error on {src_file}: {e}") # error handling

    return False


def main():
    if not input_dir.exists(): # check if input directory exists
        print(f"Error: Path '{input_dir}' does not exist.")
        return

    print("Gathering BGR images...")
    all_files = [ # collect image
        Path(root) / file
        for root, _, files in os.walk(input_dir)
        for file in files
        if os.path.splitext(file)[1] in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(all_files)} images. Applying Adaptive Median Filter...")

    tasks = [(f, input_dir, output_dir) for f in all_files] # prepare tasks for multi-threading

    with ThreadPoolExecutor() as executor: # multi-threading for faster processing
        list(
            tqdm(
                executor.map(apply_amf, tasks),
                total=len(tasks),
                desc="AMF",
            )
        )

    print(f"\nDone! AMF-filtered dataset saved to:\n{output_dir}") # success message


if __name__ == "__main__":
    main()