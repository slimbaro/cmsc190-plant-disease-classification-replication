import os
import cv2
from pathlib import Path


# paths for input and output directory
input_dir = Path("/Users/jaerish/Documents/GitHub/cmsc190-plant-disease-classification-replication/PlantVillage-Dataset/raw/color")
output_dir = Path("/Users/jaerish/Documents/GitHub/cmsc190-plant-disease-classification-replication/imagepreprocessing/resized_256x256_opencv/color")


# supported extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def resize_dataset_cv2(src: Path, dst: Path, target_size=(256, 256)):
   if not src.exists():
       print(f"Error: Path '{src}' does not exist.")
       return


   count = 0
   # walk through directory recursively to maintain plant class folder structures
   for root, _, files in os.walk(src):
       for file in files:
           ext = os.path.splitext(file)[1]
           if ext in IMAGE_EXTENSIONS:
               src_file = Path(root) / file
              
               # mirror directory structure
               relative_path = src_file.relative_to(src)
               dst_file = dst / relative_path
               dst_file.parent.mkdir(parents=True, exist_ok=True)
              
               try:
                   # read image with OpenCV
                   img = cv2.imread(str(src_file))
                  
                   if img is None:
                       print(f"Warning: Unable to read image {src_file}")
                       continue
                  
                   # resize
                   resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                  
                   # write output image
                   cv2.imwrite(str(dst_file), resized_img)
                   count += 1
               except Exception as e:
                   print(f"Error processing {src_file}: {e}")


   print(f"Done! Resized {count} images to {target_size[0]}x{target_size[1]} at:\n{dst}")


if __name__ == "__main__":
   resize_dataset_cv2(input_dir, output_dir)
