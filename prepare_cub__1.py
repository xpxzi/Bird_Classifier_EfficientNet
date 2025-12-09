import os
import shutil
import random

# ---------------- User Configuration ----------------
# Path to the extracted CUB dataset
CUB_DIR = r"C:\Users\snas2\Desktop\CUB_200_2011"

# Path where the new organized dataset will be created
DATA_DIR = r"C:\Users\snas2\Desktop\ImageClassifier\data"

# Train/validation split ratio
VAL_SPLIT = 0.1
# ---------------------------------------------------

# Create folders for train/val sets
def create_dirs(class_names):
    for subset in ["train", "val"]:
        subset_path = os.path.join(DATA_DIR, subset)
        os.makedirs(subset_path, exist_ok=True)
        for cls in class_names:
            os.makedirs(os.path.join(subset_path, cls), exist_ok=True)

# Read class names from classes.txt
def read_classes():
    classes_file = os.path.join(CUB_DIR, "classes.txt")
    class_id_name = {}
    with open(classes_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                class_id, class_name = line.split(" ", 1)
                class_id_name[int(class_id)] = class_name.replace("_", " ")
    return class_id_name

# Read image to class mapping
def read_image_labels():
    labels_file = os.path.join(CUB_DIR, "image_class_labels.txt")
    img_class_dict = {}
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                img_id, cls_id = line.split()
                img_class_dict[img_id] = int(cls_id)
    return img_class_dict

# Read image file names
def read_images():
    images_file = os.path.join(CUB_DIR, "images.txt")
    img_file_dict = {}
    with open(images_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                img_id, path = line.split()
                img_file_dict[img_id] = path
    return img_file_dict

# Copy images into train/val folders
def copy_images():
    class_id_name = read_classes()
    img_class_dict = read_image_labels()
    img_file_dict = read_images()
    create_dirs(class_id_name.values())

    for img_id, cls_id in img_class_dict.items():
        cls_name = class_id_name[cls_id]
        img_rel_path = img_file_dict[img_id]
        src_path = os.path.join(CUB_DIR, "images", img_rel_path)

        if not os.path.isfile(src_path):
            continue  # skip missing files

        # Decide train or val
        subset = "val" if random.random() < VAL_SPLIT else "train"
        dst_path = os.path.join(DATA_DIR, subset, cls_name, os.path.basename(img_rel_path))
        shutil.copy2(src_path, dst_path)

# Main
if __name__ == "__main__":
    copy_images()
    print("CUB dataset has been reorganized into train/val folders successfully.")
