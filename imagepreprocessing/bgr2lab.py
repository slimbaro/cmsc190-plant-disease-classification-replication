import sys
import subprocess

# resolving package installment error of tqdm
for package in ["tqdm", "opencv-python"]:
    try:
        __import__(package.replace("-python", ""))
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import os
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# paths for input and output directory
input_dir = Path("/Users/jaerish/Documents/GitHub/cmsc190-plant-disease-classification-replication/imagepreprocessing/resized_256x256_opencv/color")
output_dir = Path("/Users/jaerish/Documents/GitHub/cmsc190-plant-disease-classification-replication/imagepreprocessing/bgr2lab/color")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def convert_bgr_to_lab(args):
   """Worker function to read image, convert BGR -> LAB, and save."""
   src_file, src_dir, dst_dir = args
   try:
       # preserve directory structure
       relative_path = src_file.relative_to(src_dir)
       dst_file = dst_dir / relative_path
       dst_file.parent.mkdir(parents=True, exist_ok=True)


       # skip if already converted
       if dst_file.exists():
           return True


       # OpenCV reads images in BGR format by default
       img_bgr = cv2.imread(str(src_file))
      
       if img_bgr is not None:
           # convert BGR color space to CIE L*a*b*
           img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
          
           # save transformed image
           cv2.imwrite(str(dst_file), img_lab)
           return True
   except Exception as e:
       print(f"Error on {src_file}: {e}")
   return False


def main():
   if not input_dir.exists():
       print(f"Error: Path '{input_dir}' does not exist.")
       return


   print("Gathering resized images...")
   all_files = [
       Path(root) / file
       for root, _, files in os.walk(input_dir)
       for file in files
       if os.path.splitext(file)[1] in IMAGE_EXTENSIONS
   ]


   print(f"Found {len(all_files)} images. Converting BGR -> LAB...")


   tasks = [(f, input_dir, output_dir) for f in all_files]


   # process using multi-threading
   with ThreadPoolExecutor() as executor:
       list(tqdm(executor.map(convert_bgr_to_lab, tasks), total=len(tasks), desc="BGR to LAB"))


   print(f"\nDone! LAB-converted dataset saved to:\n{output_dir}")


if __name__ == "__main__":
   main()