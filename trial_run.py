# Dataset: This project uses the PlantVillage dataset
# git clone https://github.com/spMohanty/PlantVillage-Dataset.git

# show 10 pics from class
import os
from PIL import Image
import matplotlib.pyplot as plt

class_folder = "PlantVillage-Dataset/raw/color/Apple___Apple_scab"

images = os.listdir(class_folder)[:10]  # list first 10 files inside folder

# plot with 2 rows & 5 columns 
# fig for window display, width = 12in & height = 6in
fig, axes = plt.subplots(2, 5, figsize=(12, 6)) 

for ax, img_name in zip(axes.flatten(), images):
    # flattens 2D array into a 1D array for easier looping
    # zip pairs each ax with each image name

    img_path = os.path.join(class_folder, img_name) # combine folder path with image name to get full path
    img = Image.open(img_path) # open path

    ax.imshow(img) # displays image on the ax
    ax.axis("off") # removes axis labels and ticks

plt.tight_layout() # fix spacing between subplots
plt.show() # show figure