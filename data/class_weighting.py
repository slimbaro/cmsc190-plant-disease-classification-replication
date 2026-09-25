"""
Replicating "class-weighted technique" (Section 3.3)

    1. Use compute_class_weights from sklearn library to calculate weights that should be given to each class depending on the class numbers
    (NOTE: higher weight for minority classes, lower for majority classes)

    2. Use class weights as inputs to train model by setting the class weight parameters using the previously calculated weights with the fit function
    Pass weights into the Keras model's .fit(class_weight=...)
"""

import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

DATASET_DIR = "PlantVillage-Dataset/raw/color"  # path to dataset folder

# scan dataset folder function
def get_labels_from_directory(dataset_dir):

    # sort alphabetically & stored as class_names list
    class_names = sorted( 
        d for d in os.listdir(dataset_dir) # get all files/folders
        if os.path.isdir(os.path.join(dataset_dir, d)) # check if folder -> if so, keep
    )
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)} # map class names to indices (e.g., Apple___Apple_scab -> 0, Apple___Black_rot -> 1, etc.)

    labels = [] # empty list
    for class_name in class_names: # loop through each class name
        class_dir = os.path.join(dataset_dir, class_name) # get full path to class folder (e.g., PlantVillage-Dataset/raw/color/Apple___Apple_scab)
        n_images = sum( # count
            1 for f in os.listdir(class_dir) # loop through each file in class folder
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        labels.extend([class_to_idx[class_name]] * n_images) # add label (e.g. Apple___Apple_scab -> 0 & n_images = 2) then list becomes [0, 0] (2 images of Apple___Apple_scab)

    return np.array(labels), class_names, class_to_idx # results


def main():
    labels, class_names, class_to_idx = get_labels_from_directory(DATASET_DIR) # get info from dataset directory

    unique_classes = np.unique(labels) # remove duplicate

    weights = compute_class_weight(
        class_weight = "balanced", # weights given by n_samples (N) / (n_classes (C) * np.bincount(y) (ni - samples in class i)) 
        classes = unique_classes,  
        y = labels,
    )

    class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)} # convert to dictionary for Keras model.fit() function (Step 2)

    # print
    print(f"{'Class':45s} {'Count':>8s} {'Weight':>20s}")
    print("==================================================================================")
    
    total_count = 0 # for all classes count
    for cls in unique_classes: # loop through each class (for class count)
        name = class_names[cls] # get class name
        count = int(np.sum(labels == cls)) # count class samples
        total_count += count
        print(f"{name:45s} {count:8d} {class_weight_dict[cls]:25}") # print row with class name, count, and weight

    print("==================================================================================")
    print(f"{'Total':45s} {total_count:8d}")

    return class_weight_dict, class_to_idx

    ''' for checking, manual solving:
    38 classes has 54,305 images
        1) Class: Apple___Apple_scab,   
           Count:  1000 
           Weight:  = n_samples (N) / (n_classes (C) * np.bincount(y))   
                    = 54,305 / (38 * 630)   
                    = 2.268379282

                    from sklearn.utils.class_weight import compute_class_weight, 
                        it's weighted as 2.268379281537176, which is aligned with the manual calculation
    '''


if __name__ == "__main__":
    class_weight_dict, class_to_idx = main() # run main

    # Step 2 will be applied during model training
    # trialExample = model.fit(
    #     .
    #     .
    #     .
    #     class_weight = class_weight_dict,   # Balancing step
    # )