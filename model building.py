# Complete Optimized Training Solution for Fast U-Net Training

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Setup and Path Configuration
print("TensorFlow version:", tf.__version__)

# Define paths
IMG_PATH = "D:/Image Segmentation/preprocessed_images/train/images/"
MSK_PATH = "D:/Image Segmentation/Adult tooth segmentation dataset/data_split/train/masks/"

# Check if directories exist
print(f"Image path exists: {os.path.exists(IMG_PATH)}")
print(f"Mask path exists: {os.path.exists(MSK_PATH)}")

if os.path.exists(IMG_PATH):
    img_files = [f for f in os.listdir(IMG_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(img_files)} image files")

if os.path.exists(MSK_PATH):
    msk_files = [f for f in os.listdir(MSK_PATH) if f.lower().endswith(('.bmp', '.jpg', '.png'))]
    print(f"Found {len(msk_files)} mask files")

# 2. GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU")


# 3. Custom Metrics
def jaccard_index(y_true, y_pred):
    """Calculates the Jaccard index (IoU)"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return intersection / (total + tf.keras.backend.epsilon())


def dice_coefficient(y_true, y_pred):
    """Calculates the Dice coefficient"""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + tf.keras.backend.epsilon())


# 4. Optimized Data Generator
class FastDataGenerator(keras.utils.PyDataset):
    def __init__(self, img_files, mask_files, img_path, mask_path,
                 batch_size=16, size=(256, 256), seed=1, shuffle=True, **kwargs):
        super().__init__(**kwargs)

        self.img_filenames = img_files
        self.mask_filenames = mask_files
        self.img_path = img_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.size = size
        self.seed = seed
        self.shuffle = shuffle

        # Create mapping from preprocessed image names to mask names
        self.img_to_mask_map = {}
        for img_file in self.img_filenames:
            # Convert preprocessed image name back to original mask name
            original_name = img_file.replace('_preprocessed.jpg', '.bmp')
            self.img_to_mask_map[img_file] = original_name

        self.indexes = np.arange(len(self.img_filenames))

        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.img_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        images = []
        masks = []

        for idx in batch_indices:
            try:
                # Load image
                img_file = self.img_filenames[idx]
                img = tf.keras.preprocessing.image.load_img(
                    os.path.join(self.img_path, img_file),
                    color_mode='grayscale',
                    target_size=self.size
                )
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img / 255.0

                # Load corresponding mask
                mask_file = self.img_to_mask_map.get(img_file)
                if mask_file and os.path.exists(os.path.join(self.mask_path, mask_file)):
                    mask = tf.keras.preprocessing.image.load_img(
                        os.path.join(self.mask_path, mask_file),
                        color_mode='grayscale',
                        target_size=self.size
                    )
                    mask = tf.keras.preprocessing.image.img_to_array(mask)
                    mask = mask / 255.0
                    mask = (mask > 0.5).astype(np.float32)  # Binary threshold
                else:
                    # Create dummy mask if not found
                    mask = np.zeros((*self.size, 1), dtype=np.float32)
                    print(f"Warning: Mask not found for {img_file}, using dummy mask")

                images.append(img)
                masks.append(mask)

            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                # Create dummy data
                images.append(np.zeros((*self.size, 1), dtype=np.float32))
                masks.append(np.zeros((*self.size, 1), dtype=np.float32))

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# 5. Lightweight U-Net Model
def create_fast_unet(input_shape=(256, 256, 1), num_classes=1):
    """Lightweight U-Net with fewer parameters"""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(c9)

    model = models.Model(inputs, outputs)
    return model


# 6. Main Training Function
def train_optimized_model():
    print("\n=== Starting Optimized Training ===")

    # Create train/validation split
    train_img_files, val_img_files = train_test_split(
        img_files, test_size=0.2, random_state=42
    )

    print(f"Training images: {len(train_img_files)}")
    print(f"Validation images: {len(val_img_files)}")

    # Create data generators
    train_generator = FastDataGenerator(
        img_files=train_img_files,
        mask_files=[],  # Will be mapped automatically
        img_path=IMG_PATH,
        mask_path=MSK_PATH,
        batch_size=16,  # Increased batch size
        size=(256, 256),  # Reduced image size
        shuffle=True
    )

    val_generator = FastDataGenerator(
        img_files=val_img_files,
        mask_files=[],  # Will be mapped automatically
        img_path=IMG_PATH,
        mask_path=MSK_PATH,
        batch_size=16,
        size=(256, 256),
        shuffle=False
    )

    # Create model
    model = create_fast_unet()

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coefficient, jaccard_index]
    )

    # Print model info
    print(f"\nModel has {model.count_params():,} parameters")
    print(f"Training batches per epoch: {len(train_generator)}")
    print(f"Validation batches per epoch: {len(val_generator)}")

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model_fast.keras',
            monitor='val_dice_coefficient',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    import time
    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,  # Reduced epochs for faster testing
        callbacks=callbacks,
        verbose=1
    )

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nTraining completed in {training_time / 60:.2f} minutes")
    print(f"Average time per epoch: {training_time / 20:.2f} seconds")

    return model, history


# 7. Visualization Function
def plot_training_results(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Dice Coefficient
    axes[1, 0].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[1, 0].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()

    # Jaccard Index
    axes[1, 1].plot(history.history['jaccard_index'], label='Train IoU')
    axes[1, 1].plot(history.history['val_jaccard_index'], label='Val IoU')
    axes[1, 1].set_title('Jaccard Index (IoU)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


# 8. Run Training
if __name__ == "__main__":
    # Check if we have data
    if 'img_files' in locals() and len(img_files) > 0:
        print("Starting optimized training...")
        model, history = train_optimized_model()

        # Plot results
        plot_training_results(history)

        print("Training completed! Model saved as 'best_model_fast.keras'")
    else:
        print("No image files found. Please check your paths.")