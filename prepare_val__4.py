# prepare_val.py
# English comments explaining each step

import os
import shutil
import random

# ---------------- Paths ----------------
DATA_DIR = "C:\\Users\\snas2\\Desktop\\ImageClassifier\\data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
VAL_SPLIT = 0.15  # 15% of images per class to validation

# Ensure val directory exists
os.makedirs(VAL_DIR, exist_ok=True)

# Iterate through each class
for class_name in os.listdir(TRAIN_DIR):
    class_train_dir = os.path.join(TRAIN_DIR, class_name)
    class_val_dir = os.path.join(VAL_DIR, class_name)
    os.makedirs(class_val_dir, exist_ok=True)

    # List all images in class
    images = [f for f in os.listdir(class_train_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(images)
    num_val = int(len(images) * VAL_SPLIT)

    # Move selected images to val folder
    for img_name in images[:num_val]:
        src = os.path.join(class_train_dir, img_name)
        dst = os.path.join(class_val_dir, img_name)
        shutil.move(src, dst)

print("Validation set created successfully.")
