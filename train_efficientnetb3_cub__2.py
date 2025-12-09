# train_efficientnetb3_cub.py
# English comments above each logical block/line as requested.

import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- User paths (edit if needed) ----------------
# Path where data folder contains train/ and val/ subfolders
DATA_DIR = r"C:\Users\snas2\Desktop\ImageClassifier\data"
# Path where final model will be saved
MODEL_SAVE_PATH = r"C:\Users\snas2\Desktop\ImageClassifier\models\efficientnetb3_cub_best.keras"

# ---------------- Parameters ----------------
# EfficientNetB3 recommended input size is 300
IMG_SIZE = (300, 300)
BATCH_SIZE = 16              # adjust to GPU memory; lower if OOM
NUM_CLASSES = 200            # CUB-200
HEAD_EPOCHS = 10             # train classifier head first
FINE_TUNE_STAGE1_EPOCHS = 20 # unfreeze last N layers
FINE_TUNE_STAGE2_EPOCHS = 10 # optional further unfreeze (small LR)
LR_HEAD = 1e-3
LR_FINE = 3e-5
LR_FINE_LAST = 1e-6
UNFREEZE_LAST_N = 50         # unfreeze last N layers first

# ---------------- Mixed precision (optional, speeds up on modern GPUs) ----------------
# Enable mixed precision for faster training on supported GPUs
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print("[INFO] Mixed precision enabled.")
except Exception:
    # If not supported, continue without it
    print("[INFO] Mixed precision not enabled or not supported in this environment.")

# ---------------- Preprocessing ----------------
# Use EfficientNet preprocessing function
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# ---------------- Data generators ----------------
# Training augmentation (moderate, preserves class features)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12,
    brightness_range=(0.8,1.2),
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation generator (only preprocessing)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ---------------- Build model ----------------
# Load EfficientNetB3 base with ImageNet weights (exclude top)
base_model = tf.keras.applications.EfficientNetB3(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model initially
base_model.trainable = False

# Build classification head
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
# Final Dense with float32 activation to avoid fp16 softmax issues when mixed precision on
x = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = models.Model(inputs, x)

# ---------------- Compile head ----------------
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_HEAD),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

# ---------------- Callbacks ----------------
callbacks_head = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
]

# ---------------- Train head ----------------
print("[INFO] Starting head training...")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks_head,
    verbose=1
)

# ---------------- Fine-tune Stage 1: unfreeze last N layers ----------------
print(f"[INFO] Fine-tuning: unfreeze last {UNFREEZE_LAST_N} layers of the base model.")
base_model.trainable = True

# Freeze all layers except the last UNFREEZE_LAST_N layers
if UNFREEZE_LAST_N > 0:
    for layer in base_model.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False
    for layer in base_model.layers[-UNFREEZE_LAST_N:]:
        layer.trainable = True

# Re-compile with lower LR for fine-tuning
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_FINE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

callbacks_fine1 = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

print("[INFO] Starting fine-tuning stage 1...")
history_fine1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_STAGE1_EPOCHS,
    callbacks=callbacks_fine1,
    verbose=1
)

# ---------------- Fine-tune Stage 2: optional further unfreeze (full unfreeze) ----------------
# If you want to push further, unfreeze more / all layers and train with a very small LR
print("[INFO] Starting fine-tuning stage 2: unfreeze full base model for final polishing.")
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_FINE_LAST),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
)

callbacks_fine2 = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-9),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history_fine2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_STAGE2_EPOCHS,
    callbacks=callbacks_fine2,
    verbose=1
)

# ---------------- Save final model ----------------
print(f"[INFO] Saving final model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("[DONE] Training complete.")
