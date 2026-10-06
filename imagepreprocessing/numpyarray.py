"""
Convert RGB image dataset to NumPy arrays (.npz format)

Input:  imagepreprocessing/bgr2rgb/color
Output: imagepreprocessing/numpy_arrays/color

Loading numpy array: (too big if its not in npz format)
data = np.load("path_to_file.npz")
img_array = data["image"]  # returns (H, W, 3) uint8 array in RGB order
"""

import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
input_dir = SCRIPT_DIR / "bgr2rgb" / "color"
output_dir = SCRIPT_DIR / "numpy_arrays" / "color"
output_dir.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

MAX_IMAGES = 20_000

def convert_to_numpy(args):
   src_file, src_dir, dst_dir = args
   try:
       relative_path = src_file.relative_to(src_dir).with_suffix(".npz")
       dst_file = dst_dir / relative_path
       dst_file.parent.mkdir(parents=True, exist_ok=True)
       if dst_file.exists():
           return True
       img_bgr = cv2.imread(str(src_file))
       if img_bgr is not None:
           img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
           img_array = np.asarray(img_rgb, dtype=np.uint8)
           np.savez_compressed(dst_file, image=img_array)
           return True
   except Exception as e:
       print(f"Error on {src_file}: {e}")
   return False

def main():
   if not input_dir.exists():
       print(f"Error: Path '{input_dir}' does not exist.")
       return
   print("Gathering RGB images...")
   all_files = [
       Path(root) / file
       for root, _, files in os.walk(input_dir)
       for file in files
       if os.path.splitext(file)[1] in IMAGE_EXTENSIONS
   ]
   
   unprocessed_files = [
       f for f in all_files
       if not (output_dir / f.relative_to(input_dir).with_suffix(".npz")).exists()
   ]
   
   files_to_process = unprocessed_files[:MAX_IMAGES]
   
   print(f"Found {len(all_files)} images. Converting to numpy arrays...")
   print(f"Processing {len(files_to_process)} images this run...")
   
   tasks = [(f, input_dir, output_dir) for f in files_to_process]
   with ThreadPoolExecutor() as executor:
       list(
           tqdm(
               executor.map(convert_to_numpy, tasks),
               total=len(tasks),
               desc="Image to numpy",
           )
       )
   print(f"\nDone! numpy array dataset saved to:\n{output_dir}")

if __name__ == "__main__":
   main()